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

# Set download folder to project directory
DOWNLOAD_FOLDER = "data"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Download dataset
task, dataset_type, hdf5_type = "lift", "ph", "low_dim"
FileUtils.download_url(
    url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
    download_dir=DOWNLOAD_FOLDER,
)

dataset_path = os.path.join(DOWNLOAD_FOLDER, "low_dim_v141.hdf5")
assert os.path.exists(dataset_path), "Dataset download failed!"

# Read dataset
with h5py.File(dataset_path, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    print(f"Dataset contains {len(demos)} demonstrations.")
    
    # Print number of samples per demonstration
    for demo in demos:
        print(f"{demo} has {f[f'data/{demo}/actions'].shape[0]} samples")

    # Explore first demonstration
    demo_grp = f[f"data/{demos[0]}"]
    for t in range(5):
        obs_t = {k: demo_grp[f"obs/{k}"][t].tolist() for k in demo_grp["obs"]}
        act_t = demo_grp["actions"][t].tolist()
        print(f"Timestep {t}\nObservations: {json.dumps(obs_t, indent=4)}\nAction: {act_t}")
    
    # Validate next observations offset
    assert all(
        np.allclose(demo_grp[f"obs/{k}"][1:], demo_grp[f"next_obs/{k}"][:-1])
        for k in demo_grp["obs"]
    ), "Mismatch between obs and next_obs!"
    print("Observation alignment check passed.")
    
    print(f"Dones: {demo_grp['dones'][:]}\nRewards: {demo_grp['rewards'][:]}")
    print(f"Model file: {demo_grp.attrs['model_file']}")
    
    # Environment metadata
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("Environment Metadata:\n", json.dumps(env_meta, indent=4))
    
    # Initialize environment and playback demonstrations
    env = EnvUtils.create_env_from_metadata(env_meta, render=False, render_offscreen=True)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    
    video_path = os.path.join(DOWNLOAD_FOLDER, "playback.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)
    
    def playback_trajectory(demo_key):
        init_state = f[f"data/{demo_key}/states"][0]
        model_xml = f[f"data/{demo_key}"].attrs["model_file"]
        env.reset_to({"states": init_state, "model": model_xml})
        
        for action in f[f"data/{demo_key}/actions"][:]:
            env.step(action)
            video_writer.append_data(env.render(mode="rgb_array", height=512, width=512, camera_name="agentview"))
    
    for demo in demos[:5]:
        print(f"Playing back {demo}")
        playback_trajectory(demo)
    
    video_writer.close()
    print("Playback video saved at", video_path)
