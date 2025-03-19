import os
import json
import h5py
import numpy as np
import robomimic
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import imageio
import env

# Project directory
DOWNLOAD_FOLDER = "data"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

dataset_path = os.path.join(DOWNLOAD_FOLDER, "pick_and_place_2.hdf5")

# Read dataset
with h5py.File(dataset_path, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    print(f"Dataset contains {len(demos)} demonstrations.")
    
    # Print dataset structure for first demo
    demo_grp = f[f"data/{demos[0]}"]
    print(f"Keys in {demos[0]}: {list(demo_grp.keys())}")
    
    # Print number of samples per demonstration
    for demo in demos:
        try:
            actions = f[f"data/{demo}/actions"]
            print(f"{demo} has {actions.shape[0]} samples")
        except KeyError:
            print(f"Skipping {demo} (actions not found)")

    # Explore first demonstration
    demo_grp = f[f"data/{demos[0]}"]
    
    # Check the number of timesteps in 'obs'
    num_timesteps_obs = len(demo_grp["obs"])  # Number of timesteps for observations
    num_timesteps_actions = len(demo_grp["actions"])  # Number of timesteps for actions
    print(f"Number of timesteps in 'obs': {num_timesteps_obs}")
    print(f"Number of timesteps in 'actions': {num_timesteps_actions}")
    
    # Ensure both 'actions' and 'obs' have the same number of timesteps
    num_timesteps = num_timesteps_obs  # Use the number of timesteps from either
    
    # Explore actions and observations for the first demo
    for t in range(min(num_timesteps, 6)):  # Avoid going out of range
        try:
            obs_t = {k: demo_grp[f"obs/{k}"][t].tolist() for k in demo_grp["obs"]}
            act_t = demo_grp["actions"][t].tolist()
            print(f"Timestep {t}\nObservations: {json.dumps(obs_t, indent=4)}\nAction: {act_t}")
        except IndexError as e:
            print(f"IndexError at timestep {t}: {e}")
            break
    
    print(f"Dones: {demo_grp['dones'][:]}\nRewards: {demo_grp['rewards'][:]}")
    
    # Environment metadata
    env_meta = env.env["env_args"] 
    
    # Initialize environment and playback demonstrations
    env = EnvUtils.create_env_from_metadata(env_meta, render=False, render_offscreen=True)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    
    video_path = os.path.join(DOWNLOAD_FOLDER, "videos/playback.mp4")
    video_writer = imageio.get_writer(video_path, fps=10)
    
    def playback_trajectory(demo_key):
        for action in f[f"data/{demo_key}/actions"][:]:
            print(f"Action shape: {action.shape}")  # Ensure it's a 7D vector
            if np.isscalar(action):
                action = np.array([action] * 7)  # Ensure it is a 7D vector if it's scalar
            elif action.shape != (7,):
                print(f"Warning: Expected action shape (7,), got {action.shape}")
            env.step(action)
            video_writer.append_data(env.render(mode="rgb_array", height=1000, width=1520, camera_name="frontview"))

    for demo in demos[:1]:
        print(f"Playing back {demo}")
        playback_trajectory(demo)
    
    video_writer.close()
    print("Playback video saved at", video_path)
