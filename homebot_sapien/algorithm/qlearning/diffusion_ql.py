import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from homebot_sapien.algorithm.imitation.networks.image_state_policy import (
    DiffusionPolicy,
)
from homebot_sapien.algorithm.qlearning.critic import QNetwork
from homebot_sapien.algorithm.qlearning.replay_buffer import (
    DictObsReplayBuffer,
    DictObsPrioritizedReplayBuffer,
)
from typing import Union


class Diffusion_QL(object):
    def __init__(
        self,
        diffusion_config: dict,
        discount,
        tau,
        device,
        pretrained_actor: DiffusionPolicy = None,
        max_q_backup=False,
        eta=1.0,
        beta_schedule="linear",
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=1.0,
    ):

        robot_state_dim = diffusion_config["robot_state_dim"]
        action_dim = diffusion_config["action_dim"]
        chunk_size = diffusion_config["prediction_horizon"]
        if pretrained_actor is not None:
            self.actor = pretrained_actor
        else:
            self.actor = DiffusionPolicy(diffusion_config)
        self.actor.to(device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.nets.parameters(), lr=lr, weight_decay=1e-6
        )

        self.device = device
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        # self.ema = EMA(ema_decay)
        self.ema_model = self.actor.ema
        self.update_ema_every = update_ema_every

        self.critic = QNetwork(robot_state_dim, action_dim, chunk_size)
        self.critic.to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.nets.parameters(), lr=lr, weight_decay=0.0
        )

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )

        self.transformations = None
        self.expert_transformations = None
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema_model.step(self.actor.nets)

    def train_bc(
        self,
        demo_buffer: DictObsReplayBuffer,
        self_success_buffer: DictObsReplayBuffer,
        batch_size: int,
        self_ratio: float,
    ):
        metric = {
            "bc_loss": [],
            "actor_grad_norm": [],
        }
        self.actor.train()
        demo_bsz = int(batch_size * (1 - self_ratio))
        self_bsz = batch_size - demo_bsz
        for mb_idx in range(
            (len(demo_buffer) + len(self_success_buffer)) // batch_size
        ):
            if demo_bsz > 0:
                (
                    expert_state_dict,
                    expert_action,
                    _,
                    _,
                    _,
                    expert_is_pad,
                    *_,
                ) = demo_buffer.sample(demo_bsz)
                # For debugging only, pad expert actions shorter
                # expert_is_pad[:, 12:] = True
            if self_bsz > 0:
                (
                    self_state_dict,
                    self_action,
                    _,
                    _,
                    _,
                    self_is_pad,
                    *_,
                ) = self_success_buffer.sample(self_bsz)
            # if mb_idx == 10:
            #     import pickle
            #     with open("debug.pkl", "wb") as f:
            #         pickle.dump({"expert_state_dict": expert_state_dict, "self_state_dict": self_state_dict,
            #                      "expert_action": expert_action, "self_action": self_action,
            #                      "expert_is_pad": expert_is_pad, "self_is_pad": self_is_pad}, f)
            #     exit()
            # merge the batches
            key_list = (
                expert_state_dict.keys() if demo_bsz > 0 else self_state_dict.keys()
            )
            for key in key_list:
                if demo_bsz == 0:
                    expert_state_dict = self_state_dict
                    expert_action = self_action
                    expert_is_pad = self_is_pad
                elif self_bsz == 0:
                    pass
                else:
                    if isinstance(expert_state_dict[key], np.ndarray):
                        expert_state_dict[key] = np.concatenate(
                            [expert_state_dict[key], self_state_dict[key]], axis=0
                        )
                    else:
                        assert isinstance(expert_state_dict[key], list)
                        expert_state_dict[key] = (
                            expert_state_dict[key] + self_state_dict[key]
                        )
                    expert_action = np.concatenate([expert_action, self_action], axis=0)
                    expert_is_pad = np.concatenate([expert_is_pad, self_is_pad], axis=0)

            for key in expert_state_dict:
                if isinstance(expert_state_dict[key], np.ndarray):
                    expert_state_dict[key] = torch.from_numpy(
                        expert_state_dict[key]
                    ).to(self.device)
                else:
                    assert isinstance(expert_state_dict[key], list)
            if self.expert_transformations is None:
                original_size = expert_state_dict["image"].shape[-2:]
                ratio = 0.95
                self.expert_transformations = T.Compose(
                    [
                        T.RandomCrop(
                            size=[
                                int(original_size[0] * ratio),
                                int(original_size[1] * ratio),
                            ]
                        ),
                        T.Resize(original_size, antialias=True),
                        T.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                        T.ColorJitter(
                            brightness=0.3, contrast=0.4, saturation=0.5
                        ),  # , hue=0.08)
                    ]
                )
            bsz, n_image, img_size = (
                expert_state_dict["image"].shape[0],
                expert_state_dict["image"].shape[1],
                expert_state_dict["image"].shape[2:],
            )
            expert_state_dict["image"] = self.expert_transformations(
                expert_state_dict["image"].view(bsz * n_image, *img_size)
            ).view(bsz, n_image, *img_size)
            expert_action = torch.from_numpy(expert_action).float().to(self.device)
            expert_is_pad = torch.from_numpy(expert_is_pad).bool().to(self.device)
            bc_loss = self.actor.compute_bc_loss(
                expert_state_dict["image"],
                None,
                expert_state_dict["robot_state"],
                expert_action,
                expert_is_pad,
            )
            self.actor_optimizer.zero_grad()
            bc_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            else:
                actor_grad_norms = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2).to(self.device)
                            for p in self.actor.parameters()
                            if p.grad is not None
                        ]
                    ),
                    2,
                )
            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.shape)
            # exit()
            self.actor_optimizer.step()
            metric["bc_loss"].append(bc_loss.item())
            metric["actor_grad_norm"].append(actor_grad_norms.item())

            """ Step Target network """
            # In compute_bc_loss, ema is already stepped. Do not need to explicitly step_ema here

        for key in metric:
            metric[key] = np.mean(metric[key])
        return metric

    def train(
        self,
        replay_buffer: Union[DictObsReplayBuffer, DictObsPrioritizedReplayBuffer],
        n_q_iterations,
        q_only=False,
        batch_size=100,
        log_writer=None,
        augment_image=True,
        demo_buffer: DictObsReplayBuffer = None,
    ):

        metric = {
            "bc_loss": [],
            "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "q1_pred": [],
            "q2_pred": [],
            "target_q": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
        }
        for _ in range(n_q_iterations):
            # Sample replay buffer / batch
            if not isinstance(replay_buffer, DictObsPrioritizedReplayBuffer):
                (
                    state_dict,
                    action,
                    reward,
                    next_state_dict,
                    done,
                    is_pad,
                ) = replay_buffer.sample(batch_size)
                weights, sample_idxs = None, None
            else:
                (
                    state_dict,
                    action,
                    reward,
                    next_state_dict,
                    done,
                    is_pad,
                    weights,
                    sample_idxs,
                ) = replay_buffer.sample(batch_size, beta=0.4)
            # if np.any(done):
            #     debug_idx = np.where(done)[0][0]
            #     print("debug", np.linalg.norm(state_dict["image"][debug_idx][-1] - next_state_dict["image"][debug_idx][-1]))
            #     if np.linalg.norm(state_dict["image"][debug_idx][-1] - next_state_dict["image"][debug_idx][-1]) > 2000:
            #         plt.imsave("debug_obs.png", np.transpose(state_dict["image"][debug_idx][-1], (1, 2, 0)).astype(np.uint8))
            #         plt.imsave("debug_obs_tp1.png", np.transpose(next_state_dict["image"][debug_idx][-1], (1, 2, 0)).astype(np.uint8))
            #         # exit()
            for key in state_dict.keys():
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = torch.from_numpy(state_dict[key]).to(self.device)
                    next_state_dict[key] = torch.from_numpy(next_state_dict[key]).to(
                        self.device
                    )
                else:
                    assert isinstance(state_dict[key], list)
            action = torch.from_numpy(action).float().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            not_done = torch.from_numpy(1 - done).float().to(self.device)
            is_pad = torch.from_numpy(is_pad).bool().to(self.device)
            if len(reward.shape) == 1:
                reward = torch.unsqueeze(reward, dim=-1)
            if len(not_done.shape) == 1:
                not_done = torch.unsqueeze(not_done, dim=-1)

            if augment_image:
                if self.transformations is None:
                    original_size = state_dict["image"].shape[-2:]
                    ratio = 0.95
                    self.transformations = T.Compose(
                        [
                            T.RandomCrop(
                                size=[
                                    int(original_size[0] * ratio),
                                    int(original_size[1] * ratio),
                                ]
                            ),
                            T.Resize(original_size, antialias=True),
                            T.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                            T.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5
                            ),  # , hue=0.08)
                        ]
                    )
                bsz, n_image, img_size = (
                    state_dict["image"].shape[0],
                    state_dict["image"].shape[1],
                    state_dict["image"].shape[2:],
                )
                state_dict["image"] = self.transformations(
                    state_dict["image"].view(bsz * n_image, *img_size)
                ).view(bsz, n_image, *img_size)
                next_state_dict["image"] = self.transformations(
                    next_state_dict["image"].view(bsz * n_image, *img_size)
                ).view(bsz, n_image, *img_size)

            """ Q Training """
            current_q1, current_q2 = self.critic.forward(
                state_dict["robot_state"],
                state_dict["image"],
                action,
                is_pad,
            )

            if self.max_q_backup:
                # next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                # next_action_rpt = self.ema_model(next_state_rpt)
                # target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                # # buggy in dimension?
                # target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                # target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                # actor is non-deterministic, so get more predictions
                _target_q1_preds, _target_q2_preds = [], []
                self.actor.eval()
                for _ in range(10):
                    with torch.no_grad():
                        _next_action_rl_pred = self.actor.rl_pred(
                            next_state_dict["robot_state"],
                            next_state_dict["image"],
                            use_averaged_model=True,
                        )
                        _q1, _q2 = self.critic_target.forward(
                            next_state_dict["robot_state"],
                            next_state_dict["image"],
                            _next_action_rl_pred,
                        )
                        _target_q1_preds.append(_q1)
                        _target_q2_preds.append(_q2)
                target_q1 = torch.stack(_target_q1_preds, dim=-1).max(dim=-1)[0]
                target_q2 = torch.stack(_target_q2_preds, dim=-1).max(dim=-1)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                self.actor.eval()
                with torch.no_grad():
                    next_action = self.actor.rl_pred(
                        next_state_dict["robot_state"],
                        next_state_dict["image"],
                        use_averaged_model=True,
                    )
                    target_q1, target_q2 = self.critic_target.forward(
                        next_state_dict["robot_state"],
                        next_state_dict["image"],
                        next_action,
                    )
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            if weights is None:
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                    current_q2, target_q
                )
            else:
                critic_loss = (
                    0.5
                    * (
                        torch.from_numpy(weights).to(self.device)
                        * (
                            torch.square(current_q1 - target_q)
                            + torch.square(current_q2 - target_q)
                        ).squeeze(dim=-1)
                    ).mean()
                )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            else:
                critic_grad_norms = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2).to(self.device)
                            for p in self.critic.parameters()
                            if p.grad is not None
                        ]
                    ),
                    2,
                )
            self.critic_optimizer.step()
            if isinstance(replay_buffer, DictObsPrioritizedReplayBuffer):
                with torch.no_grad():
                    td_error = 0.5 * (
                        torch.square(current_q1 - target_q)
                        + torch.square(current_q2 - target_q)
                    ).squeeze(dim=-1)
                td_error = td_error.detach().cpu().numpy()
                replay_buffer.update_priorities(sample_idxs, td_error)

            metric["critic_loss"].append(critic_loss.item())
            metric["q1_pred"].append(current_q1.mean().item())
            metric["q2_pred"].append(current_q2.mean().item())
            metric["target_q"].append(target_q.mean().item())
            metric["critic_grad_norm"].append(critic_grad_norms.item())
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        """ Policy Training """
        if not q_only:
            self.actor.train()
            # bc_loss = self.actor.loss(action, state)
            if demo_buffer is not None:
                (
                    expert_state_dict,
                    expert_action,
                    _,
                    _,
                    _,
                    expert_is_pad,
                    *_,
                ) = demo_buffer.sample(batch_size)
                for key in expert_state_dict:
                    if isinstance(expert_state_dict[key], np.ndarray):
                        expert_state_dict[key] = torch.from_numpy(
                            expert_state_dict[key]
                        ).to(self.device)
                    else:
                        assert isinstance(expert_state_dict[key], list)
                if self.expert_transformations is None:
                    original_size = expert_state_dict["image"].shape[-2:]
                    ratio = 0.95
                    self.expert_transformations = T.Compose(
                        [
                            T.RandomCrop(
                                size=[
                                    int(original_size[0] * ratio),
                                    int(original_size[1] * ratio),
                                ]
                            ),
                            T.Resize(original_size, antialias=True),
                            T.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                            T.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5
                            ),  # , hue=0.08)
                        ]
                    )
                if augment_image:
                    bsz, n_image, img_size = (
                        expert_state_dict["image"].shape[0],
                        expert_state_dict["image"].shape[1],
                        expert_state_dict["image"].shape[2:],
                    )
                    expert_state_dict["image"] = self.expert_transformations(
                        expert_state_dict["image"].view(bsz * n_image, *img_size)
                    ).view(bsz, n_image, *img_size)
                expert_action = torch.from_numpy(expert_action).float().to(self.device)
                expert_is_pad = torch.from_numpy(expert_is_pad).bool().to(self.device)
                bc_loss = self.actor.compute_bc_loss(
                    expert_state_dict["image"],
                    None,
                    expert_state_dict["robot_state"],
                    expert_action,
                    expert_is_pad,
                )
            else:
                bc_loss = torch.Tensor([0]).to(self.device)
            new_action = self.actor.rl_pred(
                state_dict["robot_state"], state_dict["image"], use_averaged_model=False
            )

            q1_new_action, q2_new_action = self.critic.forward(
                state_dict["robot_state"], state_dict["image"], new_action
            )
            # why devide by another q?
            if np.random.uniform() > 0.5:
                # q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
                q_loss = -q1_new_action.mean()
            else:
                # q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
                q_loss = -q2_new_action.mean()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            else:
                actor_grad_norms = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2).to(self.device)
                            for p in self.actor.parameters()
                            if p.grad is not None
                        ]
                    ),
                    2,
                )
            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.shape)
            # exit()
            self.actor_optimizer.step()
            metric["actor_grad_norm"].append(actor_grad_norms.item())

            """ Step Target network """
            # In compute_bc_loss, ema is already stepped
            if demo_buffer is None and self.step % self.update_ema_every == 0:
                # TODO: different from diffusion-ql implementation
                self.step_ema()

        else:
            bc_loss = torch.Tensor([np.nan])
            q_loss = torch.Tensor([np.nan])
            actor_loss = torch.Tensor([np.nan])

        self.step += 1

        """ Log """
        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar(
                    "Actor Grad Norm", actor_grad_norms.max().item(), self.step
                )
                log_writer.add_scalar(
                    "Critic Grad Norm", critic_grad_norms.max().item(), self.step
                )
            log_writer.add_scalar("BC Loss", bc_loss.item(), self.step)
            log_writer.add_scalar("QL Loss", q_loss.item(), self.step)
            log_writer.add_scalar("Critic Loss", critic_loss.item(), self.step)
            log_writer.add_scalar("Target_Q Mean", target_q.mean().item(), self.step)

        metric["actor_loss"].append(actor_loss.item())
        metric["bc_loss"].append(bc_loss.item())
        metric["ql_loss"].append(q_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        for key in metric:
            metric[key] = np.mean(metric[key])
        return metric

    def sample_action(self, mb_robot_state, mb_image):
        assert mb_robot_state.shape[0] == 1
        rpt_robot_state = torch.tile(
            mb_robot_state.unsqueeze(dim=0), (32, 1, 1)
        ).reshape(-1, *mb_robot_state.shape[1:])
        rpt_image = torch.tile(mb_image.unsqueeze(dim=0), (32, 1, 1, 1, 1, 1)).reshape(
            -1, *mb_image.shape[1:]
        )
        with torch.no_grad():
            actions = self.actor.rl_pred(
                rpt_robot_state, rpt_image, use_averaged_model=True
            )
            is_pad = torch.zeros((actions.shape[0], actions.shape[1]), dtype=bool).to(
                self.device
            )
            q1, q2 = self.critic_target.forward(
                rpt_robot_state, rpt_image, actions, is_pad
            )
            q_value = torch.min(q1, q2)
            q_value = q_value.reshape((32,))
            idx = torch.Tensor([torch.argmax(q_value)]).long()
            # idx = torch.multinomial(F.softmax(q_value), num_samples=1)
            selected_action = actions[idx]
        return selected_action

    # The action space in our setting is large, need more smart samping
    # def sample_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
    #     with torch.no_grad():
    #         action = self.actor.sample(state_rpt)
    #         q_value = self.critic_target.q_min(state_rpt, action).flatten()
    #         idx = torch.multinomial(F.softmax(q_value), 1)
    #     return action[idx].cpu().data.numpy().flatten()

    def save_model(self, fname: str):
        torch.save(
            {"policy": self.actor.serialize(), "critic": self.critic.nets.state_dict()},
            fname,
        )

    def load_model(self, fname):
        checkpoint = torch.load(fname)
        self.actor.deserialize(checkpoint["policy"])
        self.critic.nets.load_state_dict(checkpoint["critic"])
        self.critic_target.nets.load_state_dict(checkpoint["critic"])
