from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Mini3_Loco_Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]

        random_default_pos = False
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "right_hip_pitch_joint": -0.4,
            "right_hip_roll_joint": -0.1,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": 0.8,
            "right_ankle_pitch_joint": -0.45,
            "right_ankle_roll_joint": 0.1,
            "left_hip_pitch_joint": -0.4,
            "left_hip_roll_joint": 0.1,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": 0.8,
            "left_ankle_pitch_joint": -0.45,
            "left_ankle_roll_joint": -0.1,
            "waist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 1.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 1.0,
        }

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast

        edge_width_thresh = 0
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.016, 0.048]
        simplify_grid = False
        gap_size = [0.016, 0.08]
        stepping_stone_distance = [0.016, 0.064]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        min_difficulty_level = 0.5
        test_mode = False
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.64, -0.56, -0.48, -0.40, -0.32, -0.24, -0.16, -0.08,
                                0., 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]
        measured_points_y = [-0.40, -0.32, -0.24, -0.16, -0.08,
                                0., 0.08, 0.16, 0.24, 0.32, 0.40]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 14
        terrain_width = 4
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 30 # number of terrain cols (types)
        
        terrain_dict = {"roughness": 1., 
                        "slope": 1.,
                        "pit": 1,
                        "gap": 1,
                        "stair": 1,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 0.6# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False
        num_goals = 8
        difficulty_level = 1
        test_mode = False
        
    class env(LeggedRobotCfg.env):
        num_observations = 72   # 9 + 3 * num_actions(21)
        num_actions = 21
        amp_motion_files = './resources/g1_amp_data/lafan_walk+run_50FPS' 
        num_amp_obs = num_actions
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        feet_info = True
        priv_info = True
        foot_force_info = True
        scan_dot = True
        num_privileged_obs = 75  # 12 + 3 * num_actions(21)
        if feet_info:
            num_privileged_obs += 12
        if priv_info:
            num_privileged_obs += 48  # 6 + 2 * num_actions(21)
        if foot_force_info:
            num_privileged_obs += 6
        if scan_dot:
            num_privileged_obs += 187

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        warp_camera = True
        warp_device = 'cuda:0'

        position = [0.0506235, 0.01753, 0.42987]
        original = (64, 64)
        resized = (64, 64)
        near_clip = 0
        far_clip = 2
        update_interval = 5
        buffer_len = 2 + 1
        fovy_range = [79.3, 79.3]

        # ----- camera randomization -----
        y_angle = [42, 48]
        z_angle = [-1, 1]
        x_angle = [-1, 1]

        rand_position = False
        x_pos_range = [-0.01, 0.01]
        y_pos_range = [-0.01, 0.01]
        z_pos_range = [-0.01, 0.01]

        dis_noise = 0

        gaussian_noise = False
        gaussian_noise_std = 0.05

        gaussian_filter = False
        gaussian_filter_kernel = [1, 3, 5]
        gaussian_filter_sigma = 1.2

        random_cam_delay = False    # randomly select tow sequential frames in the depth buffer
        # --------------------------------

        # ----- augmentation for deployment -----
        # body mask
        add_body_mask = False
        body_mask_path = './body_mask_data/body_masks_real+sim.npz'
        # crop depth
        crop_depth = False
        crop_pixels = [10, 20, 10, 5] # left top right bottom
        # ---------------------------------------


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2]
        randomize_base_mass = True
        added_mass_range = [-2., 2.]
        push_robots = True
        push_interval_s = 8; push_interval_min_s = 8
        min_push_vel_xy = 1; max_push_vel_xy = 1
        stair_no_push = False

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_com_pos = True
        com_x_pos_range = [-0.02, 0.02]
        com_y_pos_range = [-0.02, 0.02]
        com_z_pos_range = [-0.02, 0.02]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_actuation_offset = True
        actuation_offset_range = [-0.1, 0.1]

        delay_update_global_steps = 24 * 5000
        action_delay = True
        action_curr_step = [0, 1, 2]
        action_buf_len = 5

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        # Keys are matched with `if key in joint_name` (substring). Leading '*' is NOT a wildcard.
        # URDF joints look like: left_hip_pitch_joint, left_knee_pitch_joint, ...
        stiffness = {
            'hip_yaw_joint': 20.,
            'hip_roll_joint': 20.,
            'hip_pitch_joint': 30.,
            'knee_pitch_joint': 30.,
            'ankle_pitch_joint': 20.,
            'ankle_roll_joint': 20.,
            'shoulder_pitch_joint': 10,
            'shoulder_roll_joint': 10,
            'shoulder_yaw_joint': 10,
            'elbow_pitch_joint': 10,
            'waist_yaw_joint': 10,
        }  # [N*m/rad]

        damping = {
            'hip_yaw_joint': 1,
            'hip_roll_joint': 1,
            'hip_pitch_joint': 1,
            'knee_pitch_joint': 2,
            'ankle_pitch_joint': 1.0,
            'ankle_roll_joint': 1.0,
            'shoulder_pitch_joint': 1,
            'shoulder_roll_joint': 1,
            'shoulder_yaw_joint': 1,
            'elbow_pitch_joint': 1,
            'waist_yaw_joint': 1,
        } # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt =  0.002


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mini3/urdf/mini3_box.urdf'

        name = "mini3"
        foot_name = "ankle_roll"
        knee_name = "knee"
        torso_name = "waist_yaw_link"
        penalize_contacts_on = ["hip", "knee", "shoulder"]
        terminate_after_contacts_on = ["hip", "shoulder", "elbow", "knee"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        feet_indicator_offset = [[-0.04, 0, -0.035], [-0.02, 0, -0.035], [0, 0, -0.035], [0.02, 0, -0.035], [0.06, 0, -0.035],  [0.1, 0, -0.035]]
    
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        feet_min_lateral_distance_target = 0.13
        clearance_height_target = -0.5
        class scales:
            tracking_lin_vel = 2
            tracking_ang_vel = 2

            dof_acc = -5e-7
            dof_vel = -1e-3
            action_rate = -0.03
            action_smoothness = -0.05

            ang_vel_xy = -0.05
            orientation = -2.0
            joint_power = -2.5e-5
            feet_clearance = -0.25
            feet_stumble = -1.0
            torques = -1e-5
            arm_joint_deviation = -0.5
            hip_joint_deviation = -0.5
            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -1.0
            no_fly = 0.25
            feet_lateral_distance = 0.5
            feet_slippage = -0.25
            feet_contact_force = -2.5e-4
            feet_force_rate = -2.5e-4
            feet_contact_momentum = -2.5e-4
            collision = -15.
            feet_air_time = 1.0
            stuck = -1
            cheat = -2
            feet_edge = -0.5
            y_offset_pen = -0.5

        feet_contact_force_range = [160. , 480.]


class Mini3_Loco_CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunnerMulti'

    class policy:
        init_noise_std = 0.8
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        his_latent_dim = 64

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_amp = False
        amp_loader_type = 'lafan_16dof_multi'
        amp_loader_class_name = "Mini3_AMPLoader"
        
        entropy_coef = 0.01
        policy_learning_rate = 5e-4

    class runner( LeggedRobotCfgPPO.runner ):
        num_amp_frames = 5
        policy_class_name = "ActorCriticDepth"
        algorithm_class_name = "AMPPPOMulti"

        use_lerp = False
        max_iterations = 10000
        run_name = ''
        experiment_name = 'Mini3_loco'
        save_interval = 500
        amp_reward_coef = 5
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

  
