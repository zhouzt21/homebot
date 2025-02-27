import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from ..imitation.networks.base_net import ResNet18Conv, SpatialSoftmax
from ..imitation.networks.diffusion import replace_bn_with_gn


class QNetwork(nn.Module):
    def __init__(self, robot_state_dim, action_dim, chunk_size):
        super().__init__()
        self.num_images = 2
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
        self.robot_state_dim = robot_state_dim  # 10 in pose mode, 8 in joint mode
        self.ac_dim = action_dim * chunk_size  # 10 in pose mode, 8 in joint mode

        nets = []
        for i in range(2):
            backbones = []
            pools = []
            linears = []
            for _ in range(self.num_images):
                backbones.append(
                    ResNet18Conv(
                        **{
                            "input_channel": 3,
                            "pretrained": False,
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
                    torch.nn.Linear(
                        int(np.prod([self.num_kp, 2])), self.feature_dimension
                    )
                )
            backbones = nn.ModuleList(backbones)
            pools = nn.ModuleList(pools)
            linears = nn.ModuleList(linears)

            backbones = replace_bn_with_gn(backbones)  # TODO

            q_pred_net = nn.Sequential(
                nn.Linear(
                    self.feature_dimension * self.num_images
                    + self.robot_state_dim
                    + self.ac_dim,
                    256,
                ),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            nn.init.orthogonal_(q_pred_net[0].weight, np.sqrt(2))
            nn.init.zeros_(q_pred_net[0].bias)
            nn.init.orthogonal_(q_pred_net[2].weight, np.sqrt(2))
            nn.init.zeros_(q_pred_net[2].bias)
            nn.init.orthogonal_(q_pred_net[4].weight, 0.01)
            nn.init.zeros_(q_pred_net[4].bias)
            nets.append(
                nn.ModuleDict(
                    {
                        "backbones": backbones,
                        "pools": pools,
                        "linears": linears,
                        "q_pred_net": q_pred_net,
                    }
                )
            )
        self.nets = nn.ModuleList(nets)

    def forward(self, qpos, image, actions, is_pad=None):
        # qpos: (batch_size, robot_state_dim)
        # image: (batch_size, num_images, 3, h, w)
        # actions: (batch_size, chunk_length, action_dim)
        # is_pad: (batch_size, chunk_length), dtype: bool
        q_preds = []
        for i in range(2):
            net = self.nets[i]
            all_features = []
            for cam_id in range(self.num_images):
                assert len(image.shape) == 5
                cam_image = image[:, cam_id]
                cam_image = self.input_transform(cam_image / 255.0)
                cam_features = net["backbones"][cam_id](cam_image)
                pool_features = net["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = net["linears"][cam_id](pool_features)
                all_features.append(out_features)
            masked_actions = actions.clone()
            if is_pad is not None:
                masked_actions[is_pad] = 0.0
            q_input = torch.cat(
                all_features
                + [qpos]
                + [masked_actions.reshape(masked_actions.shape[0], self.ac_dim)],
                dim=1,
            )
            q_preds.append(net["q_pred_net"](q_input))
        return q_preds[0], q_preds[1]
