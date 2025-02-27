from homebot_sapien.algorithm.imitation.dataset import (
    BCDataset,
)
from homebot_sapien.algorithm.imitation.networks.image_state_policy import (
    R3MLangPolicy,
    R3MPolicy,
    MVPPolicy,
)
from homebot_sapien.algorithm.imitation.networks.robot_transformer_policy import (
    RTPolicy,
)


policy_classes = dict(
    r3mpolicy=R3MPolicy,
    mvppolicy=MVPPolicy,
    rtpolicy=RTPolicy,
)


def make_policy_from_config(config, device=None):
    # policy = R3MLangPolicy(device, n_state=8, n_images=1, hidden_size=256, n_bins=41)
    # n_action = 7
    # policy = R3MPolicy(
    #     device,
    #     n_state=7,
    #     state_proj_size=32,
    #     n_image=n_image_history,
    #     image_proj_size=256,
    #     hidden_size=256,
    #     n_bins=21,
    # )
    state_dim = 0
    robot_state_keys = config["data_config"]["robot_state_keys"]
    if "pose" in robot_state_keys:
        state_dim += 6
    if "gripper_width" in robot_state_keys:
        state_dim += 1
    if "joint" in robot_state_keys:
        state_dim += 7
    assert config["policy_name"] in policy_classes
    policy_class = policy_classes[config["policy_name"]]
    if policy_class == MVPPolicy:
        mvp_config = config["mvp_config"]
        if config["data_config"].get("image_wrist_or_head", "wrist") != "both":
            mvp_config["n_images"] = config["data_config"]["n_images"]
        else:
            mvp_config["n_images"] = config["data_config"]["n_images"] * 2
        mvp_config["state_dim"] = state_dim
        action_dim = 0
        for key in config["data_config"].get("action_keys", ("gripper", "pose")):
            if key == "is_terminate" or key == "gripper":
                action_dim += 1
            elif key == "pose":
                action_dim += 6
            elif key == "joint":
                action_dim += 7
        mvp_config["action_dim"] = action_dim
        policy = policy_class(**config["mvp_config"])
    elif policy_class == R3MPolicy:
        r3m_config = config["r3m_config"]
        r3m_config["device"] = device
        r3m_config["n_state"] = state_dim
        r3m_config["n_image"] = config["data_config"]["n_images"]
        policy = policy_class(**r3m_config)
    elif policy_class == RTPolicy:
        policy = policy_class(**config["policy_config"])
    # policy = MVPPolicy(
    #     state_dim=7, action_dim=n_action, n_images=n_image_history, use_pretrained=True
    # )

    # max_grad_norm = 0
    if device is not None:
        policy.to(device)
    return policy


def make_dataset_from_config(config, folder_name: str, file_sorted: bool = False):
    bc_dataset = BCDataset(
        folder_name=folder_name,
        n_images=config["data_config"]["n_images"],
        robot_state_keys=config["data_config"]["robot_state_keys"],
        # gripper_action_mode=config["data_config"]["gripper_action"],
        action_keys=config["data_config"].get("action_keys", ("gripper", "pose")),
        # gripper_action_scale=config["data_config"]["gripper_action_scale"],
        action_relative=config["data_config"]["action_relative"],
        image_wrist_or_head=config["data_config"].get("image_wrist_or_head", "wrist"),
        file_sorted=file_sorted,
    )
    return bc_dataset
