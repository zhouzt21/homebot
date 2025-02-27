MODE = "pose"
if MODE == "joint":
    train_config = dict(
        project_name="homebot-sim-diffusion",
        run_name="joint_datanorm_fixobsbug_100traj_scripted",
        log_dir="logs",
        policy_name="diffusionpolicy",
        demo_root="demos/scripted_opendoor_100traj_noiselevel0.5_deepgrasp_recordtargetall",
        data_config=dict(
            n_images=1,
            robot_state_keys=("joint", "gripper_width"),
            action_relative="tool",
            image_wrist_or_head="both",
            action_keys=("joint", "gripper"),
        ),
        estimate_stats=True,
        env_config=dict(
            need_door_shut=True,
        ),
        diffusion_config=dict(
            action_dim=8,
            robot_state_dim=8,
            observation_horizon=1,
            prediction_horizon=20,  # args['chunk_size'],
            num_queries=20,  # args['chunk_size'],
            num_train_timesteps=50,
            num_inference_timesteps=10,
            ema_power=0.75,
            vq=False,
            resnet_pretrained=False,
            inference_horizon=12,
            diffusion_step_embed_dim=256,
            loss_type="l1",
        ),
        optim=dict(
            base_lr=1e-4,
            weight_decay=1e-6,
            warmup_epoch=10,  #
            n_epoch=1000,  #
            batch_size=32,
            max_grad_norm=0,
            stop_epoch=500,
            augment_image=True,
            robot_state_noise=0,
        ),
        save_interval=50,
    )
elif MODE == "pose":
    train_config = dict(
        project_name="homebot-real-diffusion",
        run_name="real_20hz_gello_chunk48",
        # run_name="datanorm_real_20hz_usedesired_chunk48",
        # pretrained_checkpoint="logs/2024-01-30-11-39-35_datanorm_fixobsbug_100traj_noiselevel0.5_deepgrasp_usedesired_chunk20/bc_model_149.pt",
        pretrained_checkpoint=None,
        log_dir="logs",
        policy_name="diffusionpolicy",
        # demo_root=[
        #     "demos/scripted_opendoor_50traj_recordtargetall_topview_slow",
        #     # "demos/rollout_opendoor_100traj_240130113955_99",
        # ],
        # demo_root=[
        #     "demos/real_opendoor_20hz",
        #     # "demos/real_opendoor_20hz_rollout",
        # ],
        demo_root=[
            "demos/open_door_gello_20hz_trim",
        ],
        data_config=dict(
            n_images=1,
            robot_state_keys=("pose", "gripper_width"),
            action_relative="none",
            image_wrist_or_head="both",
            action_keys=("pose", "gripper"),
        ),
        estimate_stats=True,
        env_config=dict(
            need_door_shut=False,
            use_real=True,
        ),
        diffusion_config=dict(
            action_dim=10,
            robot_state_dim=10,
            observation_horizon=1,  # ! should be same as data_config["n_images"]
            prediction_horizon=48,  # args['chunk_size'],
            num_queries=48,  # args['chunk_size'],
            num_train_timesteps=50,
            num_inference_timesteps=10,
            ema_power=0.75,
            vq=False,
            resnet_pretrained=False,
            inference_horizon=12,
            diffusion_step_embed_dim=256,
            loss_type="l1",
        ),
        optim=dict(
            base_lr=1e-4,
            weight_decay=1e-6,
            warmup_epoch=10,  #
            n_epoch=1000,  #
            batch_size=32,
            max_grad_norm=0,
            stop_epoch=500,
            augment_image=True,
            robot_state_noise=0,
        ),
        eval_interval=50,
        save_interval=50,
    )
