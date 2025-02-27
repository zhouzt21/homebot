train_config = dict(
    run_name="trimrelease_rt_headimage_langmpnet_actrelativetool_freezeimg_trans3l",
    policy_name="rtpolicy",
    data_config=dict(
        overwrite=False,
        file_pattern=[
            # "/media/yunfei/homebot_demo/traj-2023-12-06*.pkl", # old control coke cola
            # "/media/yunfei/homebot_demo/traj-2023-12-07*.pkl", # old control coke cola
            # "/media/yunfei/homebot_demo/traj-2023-12-11*.pkl", # copilot coke cola
            "/media/yunfei/homebot_demo/traj-2023-12-15*.pkl",  # stationary pick, move and place
        ],
        n_images=6,
        gripper_action="abs_conti",
        robot_state_keys=(),
        action_keys=("is_terminate", "gripper", "pose"),
        gripper_action_scale=1.0,
        action_relative="tool",
        image_wrist_or_head="head",
        # robot_state_keys=(),
    ),
    policy_config=dict(
        freeze_image_encoder=True,
        num_bins=256,
        # lang_encoder="use-cmlm-multilingual",
        lang_encoder="all-mpnet-base-v2",
        transformer_num_layers=3,
    ),
    optim=dict(
        base_lr=1e-4,
        weight_decay=0.01,
        warmup_epoch=40,
        n_epoch=400,
        batch_size=11,
        max_grad_norm=0,
        stop_epoch=50,
    ),
    save_interval=10,
)
