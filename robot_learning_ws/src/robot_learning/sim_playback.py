import os
import json
import h5py
import numpy as np
import imageio

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import robosuite
from robosuite.environments.base import REGISTERED_ENVS
from color_sorting_env import ColorSortingEnv

# Define playback settings
DOWNLOAD_FOLDER = "data"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
dataset_path = os.path.join(DOWNLOAD_FOLDER, "pick_and_place_5.hdf5")

# Register the ColorSortingEnv with robosuite
REGISTERED_ENVS["ColorSortingEnv"] = ColorSortingEnv

# Environment metadata (structured setup)
env_metadata = {
    "env_args": {
        "env_name": "ColorSortingEnv",
        "env_version": "1.4.0",
        "model_path": "robosuite/robosuite/models/assets/arenas/table_arena.xml",
        "type": 1,
        "env_kwargs": {
            "has_renderer": True,
            "has_offscreen_renderer": True,
            "render_camera": "frontview",
            "ignore_done": False,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 5,
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
            "camera_names": ["frontview"],
            "camera_heights": 500,
            "camera_widths": 640,
            "camera_depths": False,
            "table_full_size": (0.94, 2.2, 0.05), 
            "reward_shaping": False
        }
    },
}

# Initialize environment using metadata
env = EnvUtils.create_env_from_metadata(env_metadata["env_args"], render=True, render_offscreen=True)

# Initialize observation utils
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})

# Read and analyze dataset
with h5py.File(dataset_path, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    print(f"Dataset contains {len(demos)} demonstrations.")

    # Print first demo structure
    demo_grp = f[f"data/{demos[0]}"]
    print(f"Keys in {demos[0]}: {list(demo_grp.keys())}")

    # Print the number of samples per demo
    for demo in demos:
        try:
            actions = f[f"data/{demo}/actions"]
            print(f"{demo} has {actions.shape[0]} samples")
        except KeyError:
            print(f"Skipping {demo} (actions not found)")

    # Ensure both 'actions' and 'obs' have the same number of timesteps
    num_timesteps_obs = len(demo_grp["obs"])
    num_timesteps_actions = len(demo_grp["actions"])
    print(f"Number of timesteps in 'obs': {num_timesteps_obs}")
    print(f"Number of timesteps in 'actions': {num_timesteps_actions}")

    # Check sample observations and actions
    for t in range(min(num_timesteps_obs, 6)):  # Limit to 6 timesteps
        try:
            obs_t = {k: demo_grp[f"obs/{k}"][t].tolist() for k in demo_grp["obs"]}
            act_t = demo_grp["actions"][t].tolist()
            print(f"Timestep {t}\nObservations: {json.dumps(obs_t, indent=4)}\nAction: {act_t}")
        except IndexError as e:
            print(f"IndexError at timestep {t}: {e}")
            break

    print(f"Dones: {demo_grp['dones'][:]}\nRewards: {demo_grp['rewards'][:]}")

# Video playback setup
video_path = os.path.join(DOWNLOAD_FOLDER, "videos/playback.mp4")
video_writer = imageio.get_writer(video_path, fps=10)

def playback_trajectory(demo_key):
    """
    Function to playback a trajectory from the dataset.
    """
    with h5py.File(dataset_path, "r") as f:
        for action in f[f"data/{demo_key}/actions"][:]:
            if np.isscalar(action):
                action = np.array([action] * 7)  # Ensure action is a 7D vector
            elif action.shape != (7,):
                print(f"Warning: Expected action shape (7,), got {action.shape}")

            env.step(action)
            video_writer.append_data(env.render(mode="rgb_array", height=912, width=1520, camera_name="frontview"))

# Play back the first demo
for demo in demos[:]:
    print(f"Playing back {demo}")
    # env.reset()  # Reset the environment to ensure cubes are placed correctly
    playback_trajectory(demo)

video_writer.close()
print("Playback video saved at", video_path)
