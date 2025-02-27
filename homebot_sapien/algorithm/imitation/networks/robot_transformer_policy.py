import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Optional
from .efficientnet import EfficientNet
from .image_state_policy import SentenceMpnetEncoder


class FilmConditioningLayer(nn.Module):
    def __init__(self, num_input: int, num_channels: int) -> None:
        super().__init__()
        self._projection_add = nn.Linear(num_input, num_channels)
        self._projection_mult = nn.Linear(num_input, num_channels)
        nn.init.zeros_(self._projection_add.weight)
        nn.init.zeros_(self._projection_add.bias)
        nn.init.zeros_(self._projection_mult.weight)
        nn.init.zeros_(self._projection_mult.bias)

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add[:, :, None, None]
            projected_cond_mult = projected_cond_mult[:, :, None, None]
        else:
            assert len(conv_filters.shape) == 2

        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result


class FilmEfficientEncoder(nn.Module):
    def __init__(self, conditioning_dim, freeze_encoder=False) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b3",
            width_coefficient=1.2,
            depth_coefficient=1.4,
            image_size=300,
            dropout_rate=0.3,
        )
        self.efficientnet.remove_head()
        if freeze_encoder:
            for param in self.efficientnet.parameters():
                param.requires_grad_(False)
        self.film_layers = nn.ModuleList()
        feature_sizes = self.efficientnet.get_feature_sizes()
        for i in range(len(feature_sizes)):
            self.film_layers.append(
                FilmConditioningLayer(conditioning_dim, feature_sizes[i])
            )
        self.conv1x1 = nn.Conv2d(
            feature_sizes[-1], 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        # nn.init.
        self.head_film_layer = FilmConditioningLayer(conditioning_dim, 512)
        # kernel_initializer=tf.keras.initializers.VarianceScaling()

        # self.image_transform = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        # Crop and resize to 300
        self.image_transform = T.Compose(
            [
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.CenterCrop(320),
                T.Resize(300),
            ]
        )

    def preprocess(self, images):
        """
        images: (batch, C, H, W)
        """
        res = self.image_transform(images / 255.0)
        return res

    def forward(self, images, conditioning):
        """
        images: raw image 0-255 (batch, C, H, W)
        conditioning: sentence embedding (batch, conditioning_dim)
        """
        x = self.preprocess(images)
        # Stem
        x = self.efficientnet._swish(
            self.efficientnet._bn0(self.efficientnet._conv_stem(x))
        )

        # Blocks
        for idx, block in enumerate(self.efficientnet._blocks):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.efficientnet._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # Film conditioning
            x = self.film_layers[idx].forward(x, conditioning)
        x = self.head_film_layer(self.conv1x1(x), conditioning)  # (B, 512, 9, 9)
        return x


def glorot_uniform(x: torch.Tensor):
    assert len(x.shape) == 2
    nn.init.uniform_(
        x,
        a=-math.sqrt(6 / (x.shape[0] + x.shape[1])),
        b=math.sqrt(6 / (x.shape[0] + x.shape[1])),
    )


def _maybe_dropout(rate: float = 0.0):
    if rate > 0:
        return nn.Dropout(rate)
    return lambda x, *args: x  # Does nothing to x.


class MLPBlock(nn.Module):
    def __init__(
        self, input_dim, bottleneck_dim, output_dim, dropout_rate: float = 0.1
    ):
        super().__init__()
        self._hidden_layer = nn.Linear(input_dim, bottleneck_dim)
        self._activation = nn.GELU()
        self._hidden_dropout = _maybe_dropout(dropout_rate)
        self._output_layer = nn.Linear(bottleneck_dim, output_dim)
        self._output_dropout = _maybe_dropout(dropout_rate)
        glorot_uniform(self._hidden_layer.weight)
        glorot_uniform(self._output_layer.weight)
        nn.init.normal_(self._hidden_layer.bias, std=1e-6)
        nn.init.normal_(self._output_layer.bias, std=1e-6)

    def forward(self, inputs):
        x = self._activation(self._hidden_layer(inputs))
        x = self._hidden_dropout(x)
        x = self._output_layer(x)
        x = self._output_dropout(x)
        return x


class TokenLearner(nn.Module):
    def __init__(
        self,
        input_channels,
        num_tokens,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp_block = MLPBlock(input_channels, bottleneck_dim, num_tokens)
        self.layer_norm = nn.LayerNorm(input_channels)

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        bs, c, h, w = inputs.shape
        inputs = torch.reshape(inputs, (bs, c, h * w)).transpose(1, 2)  # (bs, h * w, c)
        selected = self.layer_norm(inputs)
        selected: torch.Tensor = self.mlp_block(
            selected
        )  # Shape: [bs, h*w, num_tokens].
        selected = selected.transpose(1, 2)  # Shape: [bs, n_token, h*w].
        selected = nn.Softmax(dim=-1)(selected)
        feat = torch.matmul(selected, inputs)
        return feat  # Shape: [bs, n_token, c]


class _TransformerLayer(nn.Module):
    """A single transformer block."""

    def __init__(
        self,
        input_dim: int,
        layer_size: int = 4096,
        num_heads: int = 8,
        feed_forward_size: int = 512,
        dropout_rate: float = 0.1,
    ):
        """Creates a Transformer layer.

        Args:
        layer_size: Size of the multiple head attention layer.
        num_heads: Number of heads for the multiple head attention layer.
        feed_forward_size: Dimensionality of the feed_forward layer.
        dropout_rate: Dropout rate.
        return_attention_scores: Return attention scores.
        """
        super(_TransformerLayer, self).__init__()
        assert input_dim == feed_forward_size
        self.layernorm1 = nn.LayerNorm(input_dim, eps=1e-6)
        # NOTE: the implementation becomes different from the tensorflow version
        self.q_project = nn.Linear(input_dim, layer_size)
        self.k_project = nn.Linear(input_dim, layer_size)
        self.v_project = nn.Linear(input_dim, layer_size)
        self.mha1 = nn.MultiheadAttention(
            layer_size, num_heads, dropout_rate, batch_first=True
        )
        self.mha1.in_proj_weight.requires_grad_(False)
        self.mha1.in_proj_bias.requires_grad_(False)
        nn.init.eye_(self.mha1.in_proj_weight)
        nn.init.zeros_(self.mha1.in_proj_bias)
        self.mha1.out_proj.weight.requires_grad_(False)
        self.mha1.out_proj.bias.requires_grad_(False)
        nn.init.eye_(self.mha1.out_proj.weight)
        nn.init.zeros_(self.mha1.out_proj.bias)
        self.out_project = nn.Linear(layer_size, input_dim, bias=False)
        self.ff = nn.Linear(input_dim, feed_forward_size)
        self.layernorm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout_ff = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, dim), attention_mask: (T, T), True is no attention
        """
        x1 = self.layernorm1(x)
        q = self.q_project(x1)
        k = self.k_project(x1)
        v = self.v_project(x1)
        mha_results = self.mha1(q, k, v, attn_mask=attention_mask, need_weights=False)
        x1, weights = mha_results
        x1 = self.out_project(x1)
        x = x + x1
        y = self.layernorm2(x)
        ff_y = self.ff(y)
        ff_y = self.dropout_ff(ff_y)
        x = x + ff_y
        return x, weights


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_layers: int = 1,
        layer_size: int = 4096,
        num_heads: int = 8,
        feed_forward_size: int = 512,
        dropout_rate: float = 0.1,
        vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self._layers = nn.ModuleList(
            [
                _TransformerLayer(  # pylint: disable=g-complex-comprehension
                    input_dim=feed_forward_size,
                    layer_size=layer_size,
                    num_heads=num_heads,
                    feed_forward_size=feed_forward_size,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self._token_emb = nn.Linear(input_dim, feed_forward_size)
        self._position_emb = nn.Linear(seq_len, feed_forward_size)
        self._output_tokens = nn.Linear(feed_forward_size, vocab_size)
        # self.positions = None

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Calls the layer.

        Args:
        x: Input Tensor of shape `(B, T, dim)`.
        attention_mask: a boolean mask of shape `(T, T)`, that prevents
            attention to certain positions. The boolean mask specifies which query
            elements can attend to which key elements, 1 indicates no attention and 0
            indicates attention.

        Returns:
        x: Output Tensor of shape `(B, T, vocab_size)`.
        """

        assert x.shape[1] == self.seq_len
        batch_size = x.shape[0]

        positions = (
            torch.tile(
                torch.unsqueeze(torch.eye(self.seq_len), dim=0), [batch_size, 1, 1]
            )
            .float()
            .to(x.device)
        )

        x = self._token_emb(x)
        try:
            x += self._position_emb(positions)
        except RuntimeError as e:
            print("x", x.shape, "self.positions", positions.shape)
            print(e)
            raise RuntimeError
        scores = []

        for layer in self._layers:
            x, score = layer(x, attention_mask=attention_mask)
        if score is not None:
            scores.append(score)
        x = self._output_tokens(x)
        return x


class RTPolicy(nn.Module):
    def __init__(
        self,
        freeze_image_encoder=False,
        num_bins=256,
        lang_encoder="all-mpnet-base-v2",
        transformer_num_layers=8,
    ) -> None:
        super().__init__()
        self.lang_encoder = SentenceMpnetEncoder(
            model_path="~/Downloads/models/sentence-transformers/" + lang_encoder
        )
        for param in self.lang_encoder.parameters():
            param.requires_grad_(False)
        self.image_encoder = FilmEfficientEncoder(
            self.lang_encoder.embedding_size, freeze_image_encoder
        )
        self.token_learner = TokenLearner(512, 8)
        self.num_bins = num_bins
        self.transformer_network = TransformerNetwork(
            512,
            48,
            num_layers=transformer_num_layers,
            layer_size=128 * 8,
            num_heads=8,
            feed_forward_size=512,
            dropout_rate=0.1,
            vocab_size=self.num_bins,
        )
        self.action_dim = 8

    def forward(self, images, lang: List[str]):
        """
        images: (bs, history, c, h, w), in range 0-255
        lang: list of string of length bs
        """
        assert images.shape[0] == len(lang)
        bs, n_history = images.shape[0], images.shape[1]
        conditioning = self.lang_encoder.forward(lang).detach()
        # expand history dim
        conditioning = torch.tile(
            torch.unsqueeze(conditioning, dim=1), [1, n_history, 1]
        )  # (bs, n_history, dim)
        image_embedding = self.image_encoder.forward(
            images.reshape(bs * n_history, *images.shape[2:]),
            conditioning.reshape(bs * n_history, *conditioning.shape[2:]),
        )  # (bs * n_history, 512, 9, 9)
        image_tokens = self.token_learner.forward(
            image_embedding
        )  # (bs * n_history, 8, 512)
        image_tokens = image_tokens.reshape(
            bs, n_history * image_tokens.shape[1], image_tokens.shape[2]
        )
        # dummy_action_tokens = torch.zeros((bs, 11, image_tokens.shape[-1]), dtype=image_tokens.dtype).to(image_tokens.device)
        # obs_tokens = torch.cat([image_tokens, dummy_action_tokens], dim=1)  # (bs, 48 + 11, 512)
        obs_tokens = image_tokens
        out_tokens = self.transformer_network.forward(obs_tokens)
        # slice to get action predictions
        action_preds = out_tokens[:, : self.action_dim]  # (bs, action_dim, vocab_size)
        return action_preds

    def compute_log_prob(self, images, lang, actions):
        """
        images: (bs, history, c, h, w)
        lang: list of string
        actions: (bs, action_dim), continuous dimensions in range [-1, 1], discrete dimensions already as bin indices.
        action orders should be (is_terminate, gripper, dx, dy, dz, drx, dry, drz)
        """
        action_preds = self.forward(images, lang)
        action_dists = [
            torch.distributions.Categorical(logits=action_preds[:, i])
            for i in range(action_preds.shape[1])
        ]
        # Map actions to corresponding bins
        assert actions.shape[1] == self.action_dim
        action_bins = torch.clone(actions)
        action_bins[:, 1:] = torch.round(
            (action_bins[:, 1:] + 1) / 2 * (self.num_bins - 1)
        )
        log_probs = []
        for i in range(len(action_dists)):
            _dist = action_dists[i]
            _logprob = _dist.log_prob(
                torch.round(action_bins[:, i]).int().detach()
            ).unsqueeze(
                dim=-1
            )  # (bs, 1)
            # print("single logprob", _logprob)
            log_probs.append(_logprob)
        sum_log_prob = torch.sum(
            torch.cat(log_probs, dim=-1), dim=-1, keepdim=True
        )  # (bs, 1)
        return sum_log_prob

    def compute_bc_loss(
        self,
        image: torch.Tensor,
        lang,
        robot_state: Optional[torch.Tensor],
        action: torch.Tensor,
    ):
        logprob = self.compute_log_prob(image, lang, action)
        loss = -logprob.mean()
        return loss

    def act(
        self,
        images,
        lang,
        robot_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        action_preds = self.forward(images, lang)
        mode_dist = torch.distributions.Categorical(logits=action_preds[:, 0, :2])
        action_dists = [
            torch.distributions.Categorical(logits=action_preds[:, i])
            for i in range(1, action_preds.shape[1])
        ]
        actions = []
        # action mode is only possible as 0, 1, 2
        if deterministic:
            _action = torch.argmax(mode_dist.probs, dim=1, keepdim=True)
        else:
            _action = mode_dist.sample().unsqueeze(dim=-1)
        actions.append(_action)
        for i in range(len(action_dists)):
            _dist: torch.distributions.Categorical = action_dists[i]
            if deterministic:
                _action = torch.argmax(_dist.probs, dim=1, keepdim=True)  # (N, 1)
            else:
                _action = _dist.sample().unsqueeze(dim=-1)  # (N, 1)
            _action = _action / (self.num_bins - 1.0) * 2 - 1
            actions.append(_action)
        actions = torch.cat(actions, dim=-1)  # (N, action_dim)
        return actions


def test():
    policy = RTPolicy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    num_params = {"transformer_network": 0, "token_learner": 0, "image_encoder": 0}
    for name, param in policy.named_parameters():
        for key in num_params:
            if name.startswith(key) and param.requires_grad:
                num_params[key] += math.prod(param.shape)
    print(num_params)
    batch_size = 8
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    for i in range(10):
        images = torch.randint(0, 255, (batch_size, 6, 3, 300, 300)).to(device)
        lang = ["grasp a coke cola"] * batch_size
        actions = torch.rand((batch_size, 11)) * 2 - 1
        actions[:, 0] = torch.randint(0, 3, size=(batch_size,))
        actions = actions.to(device)
        logprob = policy.compute_log_prob(images, lang, None, actions)
        optimizer.zero_grad()
        loss = -logprob.mean()
        loss.backward()
        optimizer.step()
        print(logprob)
