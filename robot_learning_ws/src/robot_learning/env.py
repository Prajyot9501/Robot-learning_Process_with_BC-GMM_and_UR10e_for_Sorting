env = {
    "env_args": {
        "env_name": "ColorSortingEnv",
        "env_version": "1.4.0",
        "model_path":"robosuite/robosuite/models/assets/arenas/table_arena.xml",
        "type": 1,
        "env_kwargs": {
            # 1) Enable on-screen rendering for visualization
            "has_renderer": True,
            "has_offscreen_renderer": True,
            "render_camera": "frontview",
            # 2) Basic environment settings
            "ignore_done": False,
            "use_object_obs": True,
            "use_camera_obs": False,      # or True, if you want image-based observations
            "control_freq": 10,
            # 3) Controller configs (unchanged)
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2
            },
            "robots": ["UR5e"],
            # 4) Camera specs
            "camera_names": ["frontview"],    
            "camera_heights": 480,            # increase resolution if desired
            "camera_widths": 640,
            "camera_depths": False,
            "table_full_size": (0.8, 1.4, 0.05),
            "reward_shaping": False
        }
    },
}
