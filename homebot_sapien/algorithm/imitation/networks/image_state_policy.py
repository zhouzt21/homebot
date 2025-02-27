import copy
import mvp
from mvp.backbones import vit
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

try:
    from r3m import load_r3m
except ImportError:
    pass
from .resnet_backbone import ResnetBackbone
from collections import OrderedDict
from torch.distributions import Distribution
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
from .diffusion import replace_bn_with_gn, ConditionalUnet1D
from .base_net import ResNet18Conv, SpatialSoftmax

try:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.training_utils import EMAModel
except ImportError:
    pass

# class RTPolicy(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.lang_encoder
#         self.image_lang_fuser
#         self.token_learner
#         self.self_attn_decoder
#         self.action_predictor


class SentenceMpnetEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "~/Downloads/models/sentence-transformers/all-mpnet-base-v2",
    ):
        super().__init__()
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(model_path))
        self.model = AutoModel.from_pretrained(os.path.expanduser(model_path))
        self.embedding_size = self.forward(["a dummy sentence"]).shape[-1]

    def forward(self, sentences: List[str]):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # # Normalize embeddings
        # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # print(sentence_embeddings.shape)
        return sentence_embeddings  # (N, 768)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class ImageR3MEncoder(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        # self.device = device
        self.model = load_r3m("resnet50")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        # self.model.to(device)
        ## DEFINE PREPROCESSING
        self.transforms = T.Compose(
            [
                # T.Resize(256),
                # T.CenterCrop(224),
                T.Resize(224),
                T.CenterCrop(224),
                # T.CenterCrop(300),
                # T.Resize(224),
                # T.ToTensor()
            ]
        )  # ToTensor() divides by 255
        self.projecter = nn.Linear(2048, emb_dim)

    def forward(self, image: torch.Tensor):
        # image: (N, 3, H, W), uint8
        # import matplotlib.pyplot as plt
        # plt.imsave("raw_image.png", np.transpose(image[0].cpu().numpy(), [1, 2, 0]))
        preprocessed_image = self.transforms(image).reshape(-1, 3, 224, 224)
        # plt.imsave("preprocessed_image.png", np.transpose(preprocessed_image[0].cpu().numpy(), [1, 2, 0]))

        with torch.no_grad():
            embedding = self.model(
                preprocessed_image
            )  ## R3M expects image input to be [0-255] # [N, 2048]
        embedding = self.projecter(embedding)
        return embedding


class ImageMVPEncoder(nn.Module):
    def __init__(self, emb_dim, use_pretrained=True) -> None:
        super().__init__()
        model_name = "vitb-mae-egosoup"
        model_func = vit.vit_b16
        img_size = 256 if "-256-" in model_name else 224
        if use_pretrained:
            pretrain_path = os.path.join(
                "/tmp/mvp-download-cache", "mae_pretrain_egosoup_vit_base.pth"
            )
        else:
            pretrain_path = "none"
        self.backbone, gap_dim = model_func(pretrain_path, img_size=img_size)
        if use_pretrained:
            self.backbone.freeze()
            self.freeze = True
        else:
            self.freeze = False
        self.projector = nn.Linear(gap_dim, emb_dim)
        self.im_mean = np.array([0.485, 0.456, 0.406])
        self.im_std = np.array([0.229, 0.224, 0.225])
        # Crop view
        self.transforms = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
            ]
        )
        # Full view
        # self.transforms = T.Compose(
        #     [
        #         T.CenterCrop(320),
        #         T.Resize(224),
        #     ]
        # )

    def forward_feat_norm(self, x):
        x = self.transforms(x).reshape(-1, 3, 224, 224)
        x = (
            x / 255.0
            - torch.from_numpy(self.im_mean).float().to(x.device).view(1, 3, 1, 1)
        ) / torch.from_numpy(self.im_std).float().to(x.device).view(1, 3, 1, 1)
        feat = self.backbone.extract_feat(x)
        return self.backbone.forward_norm(feat)

    def forward_projector(self, x):
        return self.projector(x)

    def forward(self, x):
        x = self.transforms(x).reshape(-1, 3, 224, 224)
        x = (
            x / 255.0
            - torch.from_numpy(self.im_mean).float().to(x.device).view(1, 3, 1, 1)
        ) / torch.from_numpy(self.im_std).float().to(x.device).view(1, 3, 1, 1)
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat))

    def forward_feat_before_norm(self, x):
        x = self.transforms(x).reshape(-1, 3, 224, 224)
        x = (
            x / 255.0
            - torch.from_numpy(self.im_mean).float().to(x.device).view(1, 3, 1, 1)
        ) / torch.from_numpy(self.im_std).float().to(x.device).view(1, 3, 1, 1)
        feat = self.backbone.extract_feat(x)
        return feat

    def forward_feat_learnable(self, feat):
        return self.projector(self.backbone.forward_norm(feat))


class R3MPolicy(nn.Module):
    def __init__(
        self,
        device,
        n_state: int,
        state_proj_size: int,
        n_image: int,
        image_proj_size: int,
        hidden_dims: List[int] = [256, 128, 64],
        predictor_type: str = "continuous",
        n_bins: Optional[List] = None,
    ):
        super().__init__()
        # TODO: image history
        self.image_encoder = ImageR3MEncoder(device)
        self.image_feat_projector = nn.Sequential(
            # nn.LayerNorm(self.image_encoder.embedding_size),
            nn.Linear(self.image_encoder.embedding_size, image_proj_size)
        )
        self.n_image = n_image
        self.state_projector = nn.Linear(n_state, state_proj_size)
        # self.fuser = nn.Sequential(
        #     nn.LayerNorm(image_proj_size * n_image + state_proj_size),
        #     nn.Linear(image_proj_size * n_image + state_proj_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        # )
        actor_hidden_dim = hidden_dims
        activation = nn.SELU()
        actor_layers = []
        # actor_layers.append(nn.LayerNorm(emb_dim * (n_images + 1 * (state_dim > 0))))
        actor_layers.append(
            nn.Linear(
                image_proj_size * n_image + state_proj_size * (n_state > 0),
                actor_hidden_dim[0],
            )
        )
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                pass
                # actor_layers.append(nn.Linear(actor_hidden_dim[li], action_dim))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.fuser = nn.Sequential(*actor_layers)

        self.n_bins = n_bins
        if predictor_type == "discrete":
            self.action_predictor = ActionDiscretePredictor(
                actor_hidden_dim[-1], n_bins
            )
        self.action_predictor = nn.ModuleList(
            [
                # nn.Linear(hidden_size, 2),  # terminate
                nn.Linear(hidden_size, 2),  # gripper
                nn.Linear(hidden_size, n_bins),  # logits for x
                nn.Linear(hidden_size, n_bins),  # y
                nn.Linear(hidden_size, n_bins),  # z
                nn.Linear(hidden_size, n_bins),  # logits for rot x
                nn.Linear(hidden_size, n_bins),  # y
                nn.Linear(hidden_size, n_bins),  # z
            ]
        )
        actor_weights = [np.sqrt(2)] * 2
        self.init_weights(self.fuser, actor_weights, "orthogonal")
        self.init_weights(self.action_predictor, [0.01] * 7, "orthogonal")

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, robot_state: torch.Tensor):
        assert len(image.shape) == 5  # batch, n_image, C, H, W
        assert image.shape[1] == self.n_image
        with torch.no_grad():
            image_feat = self.image_encoder(
                image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
            )
        state_feat = self.state_projector(robot_state)
        image_feat = self.image_feat_projector(image_feat)
        image_feat = image_feat.reshape(
            (image.shape[0], self.n_image * image_feat.shape[1])
        )
        fused_feat = self.fuser(torch.cat([image_feat, state_feat], dim=1))
        # is_terminate_logit = self.action_predictor[0](fused_feat)
        gripper_logit = self.action_predictor[0](fused_feat)
        pose_logit = [self.action_predictor[i](fused_feat) for i in range(1, 7)]
        all_dist = [
            # Categorical(logits=is_terminate_logit),
            Categorical(logits=gripper_logit),
            *[Categorical(logits=pose_logit[i]) for i in range(len(pose_logit))],
        ]
        return all_dist

    def compute_log_prob(
        self,
        image: torch.Tensor,
        lang: List,
        robot_state: torch.Tensor,
        action: torch.Tensor,
    ):
        assert action.shape[-1] == 7
        action = torch.clone(action)
        # process action
        action[..., 1:] = (action[..., 1:] + 1) / 2 * (self.n_bins - 1)
        action_dists = self.forward(image, robot_state)
        log_probs = []
        for i in range(len(action_dists)):
            _dist = action_dists[i]
            _logprob = _dist.log_prob(
                torch.round(action[:, i]).int().detach()
            ).unsqueeze(
                dim=-1
            )  # (N, 1)
            # print("single logprob", _logprob)
            log_probs.append(_logprob)
        sum_log_prob = torch.sum(torch.cat(log_probs, dim=-1), dim=-1, keepdim=True)
        return sum_log_prob

    def compute_bc_loss(self, image, lang, robot_state, action):
        logprob = self.compute_log_prob(image, lang, robot_state, action)
        return -logprob.mean()

    def act(self, image: torch.Tensor, lang, robot_state, deterministic: bool = False):
        action_dists = self.forward(image, robot_state)
        actions = []
        for i in range(len(action_dists)):
            _dist: Categorical = action_dists[i]
            if deterministic:
                _action = torch.argmax(_dist.probs, dim=1, keepdim=True)  # (N, 1)
            else:
                _action = _dist.sample().unsqueeze(dim=-1)  # (N, 1)
            if i < 1:
                _action = _action.float()
            else:
                # scale to [-1, 1]
                _action = _action / (self.n_bins - 1.0) * 2 - 1
            actions.append(_action)
        actions = torch.cat(actions, dim=-1)  # (N, 7)
        return actions


class MVPPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        enc_type="mvp",
        n_images=1,
        emb_dim=128,
        lang_emb_dim=128,
        state_emb_dim=128,
        hidden_dims=[256, 128, 64],
        use_pretrained=True,
        predictor_type="continuous",
        num_bins: Optional[List] = None,
        lang_cond=False,
        fuse_ln=False,
        dropout=0.0,
        critic_hidden_dims=[256, 128, 64],
    ):
        super(MVPPolicy, self).__init__()

        # Encoder
        if enc_type == "mvp":
            self.image_enc = ImageMVPEncoder(
                emb_dim=emb_dim, use_pretrained=use_pretrained
            )
        elif enc_type == "resnet":
            self.image_enc = ResnetBackbone(emb_dim=emb_dim)
        # elif enc_type == "r3m":
        #     self.image_enc = ImageR3MEncoder(emb_dim=emb_dim)
        if lang_cond:
            self.lang_encoder = SentenceMpnetEncoder()
            self.lang_projector = nn.Linear(
                self.lang_encoder.embedding_size, lang_emb_dim
            )
            for param in self.lang_encoder.parameters():
                param.requires_grad_(False)
        else:
            self.lang_encoder = None
        if state_dim > 0:
            self.state_enc = nn.Linear(state_dim, state_emb_dim)
        else:
            self.state_enc = None
        self.image_dropout = nn.Dropout(dropout)
        self.state_dropout = nn.Dropout(dropout)
        self.lang_dropout = nn.Dropout(dropout)

        # Policy
        actor_hidden_dim = hidden_dims
        activation = nn.SELU()
        actor_layers = []
        actor_input_dim = (
            emb_dim * n_images
            + state_emb_dim * (state_dim > 0)
            + lang_emb_dim * lang_cond
        )
        if fuse_ln:
            actor_layers.append(nn.LayerNorm(actor_input_dim))
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                pass
                # actor_layers.append(nn.Linear(actor_hidden_dim[li], action_dim))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(self.image_enc)
        print(self.state_enc)
        print(self.actor)
        # Initialize the actor weights
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        # actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights, "orthogonal")

        self.predictor_type = predictor_type
        if predictor_type == "continuous":
            self.final_predictor = nn.Sequential(
                nn.Linear(actor_hidden_dim[-1], action_dim)
            )
            self.init_weights(self.final_predictor, [0.01], "orthogonal")
        else:
            assert len(num_bins) == action_dim
            self.final_predictor = ActionDiscretePredictor(
                actor_hidden_dim[-1], num_bins
            )

        # Critic
        critic_input_dim = actor_input_dim
        critic_layers = []
        critic_layers.append(nn.LayerNorm(critic_input_dim))
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dims)):
            if li == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[li], critic_hidden_dims[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dims)
        critic_weights.append(1.0)
        self.init_weights(self.critic, critic_weights, "orthogonal")

        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward_feat(self, image, lang, states, is_raw_image: bool = True):
        if is_raw_image:
            assert len(image.shape) == 5
            image_emb = self.image_enc(
                image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
            )
        else:
            # `image` is actually pre-computed frozen features
            frozen_image_emb = image.reshape((image.shape[0] * image.shape[1], -1))
            image_emb = self.image_enc.forward_feat_learnable(frozen_image_emb)
        image_emb = self.image_dropout(image_emb)
        joint_emb = image_emb.reshape(image.shape[0], -1)
        if self.lang_encoder is not None:
            lang_embed = self.lang_encoder(lang).detach()
            lang_embed = self.lang_projector(lang_embed)
            lang_embed = self.lang_dropout(lang_embed)
            joint_emb = torch.cat([joint_emb, lang_embed], dim=1)
        if self.state_enc is not None:
            state_emb = self.state_enc(states)
            state_emb = self.state_dropout(state_emb)
            joint_emb = torch.cat([joint_emb, state_emb], dim=1)
        # actions_feat = self.actor(joint_emb)
        return joint_emb

    def forward(self, image, lang, states, is_raw_image: bool = True):
        joint_emb = self.forward_feat(image, lang, states, is_raw_image)
        actions_feat = self.actor(joint_emb)
        actions = self.final_predictor(actions_feat)
        return actions

    def compute_bc_loss(
        self, image: torch.Tensor, lang, robot_state: torch.Tensor, action: torch.Tensor
    ):
        joint_emb = self.forward_feat(image, lang, robot_state)
        actions_feat = self.actor(joint_emb)
        if self.predictor_type == "continuous":
            pred_actions = self.final_predictor(actions_feat)
            loss = nn.MSELoss()(pred_actions, action)
        else:
            logprob = self.final_predictor.compute_log_prob(actions_feat, action)
            loss = -logprob.mean()
        return loss

    def inference(self, image, lang, robot_state, deterministic):
        joint_emb = self.forward_feat(image, lang, robot_state)
        actions_feat = self.actor(joint_emb)
        if self.predictor_type == "continuous":
            pred_actions = self.final_predictor(actions_feat)
        else:
            pred_actions = self.final_predictor.act(actions_feat, deterministic)
        return pred_actions

    # for rl efficient saving and rl computing
    def encode_image_frozen(self, image):
        frozen_feat = self.image_enc.forward_feat_before_norm(
            image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
        ).reshape(
            (image.shape[0], image.shape[1], -1)
        )  # batch_size, n_history, 768
        assert frozen_feat.shape[-1] == 768
        return frozen_feat

    def _forward_feat_rl(self, obs: dict):
        frozen_image_feat = obs["image_frozen_feat"]
        lang = obs["lang"]
        robot_state = obs["state"]
        joint_emb = self.forward_feat(
            frozen_image_feat, lang, robot_state, is_raw_image=False
        )
        return joint_emb

    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        # obs: dictionary of batched observations
        joint_emb = self._forward_feat_rl(obs)
        value = self.critic(joint_emb)
        actions_feat = self.actor(joint_emb)
        if self.predictor_type == "continuous":
            action = self.final_predictor(actions_feat)
            # TODO: add gaussian distribution
            raise NotImplementedError
        else:
            action = self.final_predictor.act(actions_feat, deterministic)
            action_log_prob = self.final_predictor.compute_log_prob(
                actions_feat, action
            )
        return value, action, action_log_prob, rnn_hxs

    def get_value(self, obs, rnn_hxs=None, rnn_masks=None):
        joint_emb = self._forward_feat_rl(obs)
        value = self.critic(joint_emb)
        return value

    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        joint_emb = self._forward_feat_rl(obs)
        actions_feat = self.actor(joint_emb)
        if self.predictor_type == "continuous":
            raise NotImplementedError
        else:
            action_log_probs = self.final_predictor.compute_log_prob(
                actions_feat, actions
            )
            dist_entropy = self.final_predictor.compute_entropy(actions_feat)
        return action_log_probs, dist_entropy, rnn_hxs


class ActionDiscretePredictor(nn.Module):
    def __init__(self, hidden_size, bin_sizes: list) -> None:
        super().__init__()
        self.action_predictor = nn.ModuleList(
            [nn.Linear(hidden_size, bin_sizes[i]) for i in range(len(bin_sizes))]
        )
        self.bin_sizes = bin_sizes
        self.init_weights(self.action_predictor, [0.01] * len(bin_sizes), "orthogonal")

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        dist_logit = [
            self.action_predictor[i](x) for i in range(len(self.action_predictor))
        ]
        all_dist = [Categorical(logits=dist_logit[i]) for i in range(len(dist_logit))]
        return all_dist

    def compute_log_prob(self, x, action):
        action = torch.clone(action)
        # process action
        action = (
            (action + 1)
            / 2
            * (torch.from_numpy(np.array(self.bin_sizes)).to(action.device) - 1)
        )
        action_dists = self.forward(x)
        log_probs = []
        for i in range(len(action_dists)):
            _dist = action_dists[i]
            _logprob = _dist.log_prob(
                torch.round(action[:, i]).int().detach()
            ).unsqueeze(
                dim=-1
            )  # (N, 1)
            # print("single logprob", _logprob)
            log_probs.append(_logprob)
        sum_log_prob = torch.sum(torch.cat(log_probs, dim=-1), dim=-1, keepdim=True)
        return sum_log_prob

    def act(self, x, deterministic: bool):
        action_dists = self.forward(x)
        actions = []
        for i in range(len(action_dists)):
            _dist: Categorical = action_dists[i]
            if deterministic:
                _action = torch.argmax(_dist.probs, dim=1, keepdim=True)  # (N, 1)
            else:
                _action = _dist.sample().unsqueeze(dim=-1)  # (N, 1)
            _action = _action / (self.bin_sizes[i] - 1.0) * 2 - 1
            actions.append(_action)
        actions = torch.cat(actions, dim=-1)  # (N, n_action)
        return actions

    def compute_entropy(self, x):
        action_dists = self.forward(x)
        entropy = torch.sum(
            torch.stack([dist.entropy() for dist in action_dists], dim=-1), dim=-1
        )
        print("In [compute_entropy], entropy shape", entropy.shape)
        return entropy


class R3MLangPolicy(nn.Module):
    def __init__(
        self, device, n_state: int, n_images: int, hidden_size: int, n_bins: int
    ):
        # Use universal sentence encoder to process the language
        # Use R3M to encode the image
        # Then fused with robot state and predict the next waypoint
        super().__init__()
        self.lang_encoder = SentenceMpnetEncoder()
        self.image_encoder = ImageR3MEncoder(device)
        self.lang_feat_projector = nn.Linear(self.lang_encoder.embedding_size, 256)
        self.image_feat_projector = nn.Linear(self.image_encoder.embedding_size, 256)
        self.state_encoder = nn.Linear(n_state, 256)
        self.n_images = n_images
        self.fuser = nn.Sequential(
            nn.Linear(n_images * 256 + 256 + 256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.n_bins = n_bins
        self.action_predictor = nn.ModuleList(
            [
                nn.Linear(hidden_size, 1),  # classify is_terminate
                nn.Linear(hidden_size, 1),  # classify gripper move
                nn.Linear(
                    hidden_size, n_bins
                ),  # logit for eef relative x. will discretize [-1, 1] into resolution of 0.025
                nn.Linear(hidden_size, n_bins),  # logits for y
                nn.Linear(hidden_size, n_bins),  # logits for z
                nn.Linear(hidden_size, n_bins),  # roll
                nn.Linear(hidden_size, n_bins),  # pitch
                nn.Linear(hidden_size, n_bins),  # yaw
            ]
        )

    def forward(
        self, image: torch.Tensor, lang: List[str], robot_state: torch.Tensor
    ) -> Dict[str, Distribution]:
        assert (
            image.shape[0] == self.n_images * len(lang)
            and image.shape[0] == self.n_images * robot_state.shape[0]
        )
        assert image.dtype == torch.uint8
        batch_size = robot_state.shape[0]
        with torch.no_grad():
            image_feat = self.image_encoder(image).reshape(
                (batch_size, self.n_images, self.image_encoder.embedding_size)
            )
            lang_feat = self.lang_encoder(lang)
            # print("image feat", image_feat.mean(), image_feat.std(), torch.norm(image_feat, p=2, dim=-1),
            #       "lang feat", lang_feat.mean(), lang_feat.std(), torch.norm(lang_feat, p=2, dim=-1))
        image_feat = self.image_feat_projector(image_feat)
        lang_feat = self.lang_feat_projector(lang_feat)
        state_feat = self.state_encoder(robot_state)

        feat = torch.cat(
            [image_feat.reshape(batch_size, -1), lang_feat, state_feat], dim=-1
        )
        fused_feat = self.fuser(feat)
        is_terminate_logit = self.action_predictor[0](fused_feat)
        gripper_logit = self.action_predictor[1](fused_feat)
        pose_logits = [self.action_predictor[i](fused_feat) for i in range(2, 8)]
        is_terminate_dist = Bernoulli(logits=is_terminate_logit)
        gripper_dist = Bernoulli(logits=gripper_logit)
        pose_dists = [
            Categorical(logits=pose_logits[i]) for i in range(len(pose_logits))
        ]
        all_dists = OrderedDict(
            {
                "is_terminate": is_terminate_dist,
                "gripper": gripper_dist,
                "eef_x": pose_dists[0],
                "eef_y": pose_dists[1],
                "eef_z": pose_dists[2],
                "eef_roll": pose_dists[3],
                "eef_pitch": pose_dists[4],
                "eef_yaw": pose_dists[5],
            }
        )
        return all_dists

    def compute_log_prob(
        self,
        image: torch.Tensor,
        lang: List[str],
        robot_state: torch.Tensor,
        action: torch.Tensor,
    ):
        """
        action[..., :2] should be 0 or 1, action[..., 2:] should lie in [-1, 1]
        assume each action dimension is independent
        """
        assert action.shape[-1] == 8
        action = torch.clone(action)
        # process action
        action[..., 2:] = (action[..., 2:] + 1) / 2 * (self.n_bins - 1)

        action_dists = self.forward(image, lang, robot_state)
        log_probs = []
        keys = action_dists.keys()
        for i, act_key in enumerate(keys):
            _dist = action_dists[act_key]
            if isinstance(_dist, Bernoulli):
                _logprob = _dist.log_prob(action[:, i : i + 1].bool().float())  # (N, 1)
                # print("single logprob", _logprob)
                log_probs.append(_logprob)
            elif isinstance(_dist, Categorical):
                _logprob = _dist.log_prob(
                    torch.round(action[:, i]).int().detach()
                ).unsqueeze(
                    dim=-1
                )  # (N, 1)
                # print("single logprob", _logprob)
                log_probs.append(_logprob)
            else:
                raise RuntimeError
        sum_log_prob = torch.sum(torch.cat(log_probs, dim=-1), dim=-1, keepdim=True)
        return sum_log_prob

    def act(
        self,
        image: torch.Tensor,
        lang: List[str],
        robot_state: torch.Tensor,
        deterministic: bool = False,
    ):
        action_dists = self.forward(image, lang, robot_state)
        actions = []
        for act_key in action_dists:
            if isinstance(action_dists[act_key], Bernoulli):
                _dist: Bernoulli = action_dists[act_key]
                if deterministic:
                    _action = (_dist.probs > 0.5).float()  # (N, 1)
                else:
                    _action = _dist.sample().float()  # (N, 1)
                actions.append(_action)
            elif isinstance(action_dists[act_key], Categorical):
                _dist: Categorical = action_dists[act_key]
                if deterministic:
                    _action = torch.argmax(_dist.probs, dim=1, keepdim=True)  # (N, 1)
                else:
                    _action = _dist.sample().unsqueeze(dim=-1)  # (N, 1)
                # scale to [-1, 1]
                _action = _action / (self.n_bins - 1.0) * 2 - 1
                actions.append(_action)
            else:
                raise RuntimeError
        actions = torch.cat(actions, dim=-1)  # (N, 8)
        return actions


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        self.num_images = 2  # number of cameras
        self.robot_state_dim = args_override.get("robot_state_dim", 8)
        self.observation_horizon = args_override[
            "observation_horizon"
        ]  ### TODO TODO TODO DO THIS
        # self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override["prediction_horizon"]  # chunk size
        self.num_inference_timesteps = args_override["num_inference_timesteps"]
        self.ema_power = args_override["ema_power"]
        self.resnet_pretrained = args_override.get("resnet_pretrained", True)
        # self.lr = args_override['lr']
        # self.weight_decay = 0

        self.input_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override["action_dim"]  # 14 + 2
        self.obs_dim = (
            self.feature_dimension * self.num_inference_timesteps + self.robot_state_dim
        )  # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in range(self.num_images):
            # ? pretrained
            backbones.append(
                ResNet18Conv(
                    **{
                        "input_channel": 3,
                        "pretrained": self.resnet_pretrained,
                        "input_coord_conv": False,
                    }
                )
            )
            pools.append(
                SpatialSoftmax(
                    **{
                        # 'input_shape': [512, 15, 20], # for 480 * 640 image
                        "input_shape": [512, 7, 7],  # for 224 * 224 image
                        "num_kp": self.num_kp,
                        "temperature": 1.0,
                        "learnable_temperature": False,
                        "noise_std": 0.0,
                    }
                )
            )
            linears.append(
                torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
            )
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        backbones = replace_bn_with_gn(backbones)  # TODO

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            # global_cond_dim=self.obs_dim*self.observation_horizon
            global_cond_dim=self.feature_dimension
            * self.num_images
            * self.observation_horizon
            + self.robot_state_dim,
            diffusion_step_embed_dim=args_override.get("diffusion_step_embed_dim", 256),
        )

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "backbones": backbones,
                        "pools": pools,
                        "linears": linears,
                        "noise_pred_net": noise_pred_net,
                    }
                )
            }
        )

        # nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=args_override.get("num_train_timesteps", 50),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

        self.loss_type = args_override.get("loss_type", "l2")

    def to(self, device: torch.device):
        super().to(device)
        self.ema.averaged_model.to(device)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        image = image.reshape(
            B * image.shape[1] // self.num_images, self.num_images, *image.shape[2:]
        )
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(self.num_images):
                assert len(image.shape) == 5
                cam_image = image[:, cam_id]
                cam_image = self.input_transform(cam_image / 255.0)
                cam_features = nets["policy"]["backbones"][cam_id](cam_image)
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                out_features = out_features.reshape(
                    B, out_features.shape[0] // B * out_features.shape[1]
                )
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=obs_cond.device,
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps
            )  # (B, chunk_size, ac_dim)

            # predict the noise residual
            noise_pred = nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond
            )

            # L2 loss
            if self.loss_type == "l2":
                all_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
            elif self.loss_type == "l1":
                all_loss = nn.L1Loss(reduction="none").forward(noise_pred, noise)
            if is_pad is not None:
                loss = (all_loss * ~is_pad.unsqueeze(-1)).mean()
            else:
                loss = all_loss.mean()

            # loss_dict = {}
            # loss_dict['l2_loss'] = loss
            # loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss
        else:  # inference time
            # To = self.observation_horizon
            # Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            all_features = []
            for cam_id in range(self.num_images):
                cam_image = image[:, cam_id]
                cam_image = self.input_transform(cam_image / 255.0)
                cam_features = nets["policy"]["backbones"][cam_id](cam_image)
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                out_features = out_features.reshape(
                    B, out_features.shape[0] // B * out_features.shape[1]
                )
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            return naction

    def rl_pred(self, qpos, image, use_averaged_model: bool):
        # for q learning, since inference is done with original net and ema will not be updated in every step
        # add post processing to generated action
        B = qpos.shape[0]
        image = image.reshape(
            B * image.shape[1] // self.num_images, self.num_images, *image.shape[2:]
        )
        Tp = self.prediction_horizon
        action_dim = self.ac_dim

        if use_averaged_model:
            nets = self.ema.averaged_model
        else:
            nets = self.nets

        all_features = []
        for cam_id in range(self.num_images):
            cam_image = image[:, cam_id]
            cam_image = self.input_transform(cam_image / 255.0)
            cam_features = nets["policy"]["backbones"][cam_id](cam_image)
            pool_features = nets["policy"]["pools"][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets["policy"]["linears"][cam_id](pool_features)
            out_features = out_features.reshape(
                B, out_features.shape[0] // B * out_features.shape[1]
            )
            all_features.append(out_features)

        obs_cond = torch.cat(all_features + [qpos], dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        naction = self._post_process_action(naction)
        return naction

    def _post_process_action(self, action):
        B = action.shape[0]
        raw_rot6d = action[:, :, 3:9].reshape(B, -1, 3, 2)
        a1, a2 = raw_rot6d[..., 0], raw_rot6d[..., 1]
        b1 = a1 / torch.norm(a1, dim=-1, keepdim=True)
        a2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = a2 / torch.norm(a2, dim=-1, keepdim=True)
        normalized_action = torch.cat(
            [
                action[..., :3],
                torch.stack([b1, b2], dim=-1).reshape(B, -1, 6),
                action[..., 9:],
            ],
            dim=-1,
        )
        return normalized_action

    def compute_bc_loss(
        self,
        image: torch.Tensor,
        lang,
        robot_state: torch.Tensor,
        action: torch.Tensor,
        is_pad: torch.Tensor = None,
    ):
        loss = self.__call__(robot_state, image, action, is_pad)
        return loss

    def inference(self, image, lang, robot_state, deterministic):
        action = self.__call__(robot_state, image)
        return action

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict()
            if self.ema is not None
            else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print("Loaded model")
        if model_dict.get("ema", None) is not None:
            print("Loaded EMA")
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = R3MLangPolicy(device, n_state=9, n_images=1, hidden_size=512, n_bins=81)
    policy.to(device)
    test_image = Image.open(
        "/home/yunfei/projects/mobile_robot/sandbox/rt-1-x/test_image.png"
    )
    img_tensor = T.PILToTensor()(test_image)[:3]
    img_tensor = torch.tile(torch.unsqueeze(img_tensor, dim=0), (1, 1, 1, 1))
    lang_instruction = ["open the door"] * 1
    robot_state = (
        torch.from_numpy(np.random.uniform(-1, 1, size=(1, 9))).float().to(device)
    )
    dists = policy.forward(img_tensor, lang_instruction, robot_state)
    for key in dists:
        print(key, dists[key].probs.shape, dists[key].sample().shape)
    test_action = torch.rand(size=(1, 8)).float().to(device)
    test_action[:, :2] = (test_action[:, :2] > 0).float()
    logprob = policy.compute_log_prob(
        img_tensor, lang_instruction, robot_state, test_action
    )
    print(logprob)
    pred_action = policy.act(img_tensor, lang_instruction, robot_state)
    print("pred_action", pred_action)
    # determ_action = policy.act(img_tensor, lang_instruction, robot_state, deterministic=True)
    # print("determ action", determ_action)


if __name__ == "__main__":
    test()
