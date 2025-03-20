# -*- coding: utf-8 -*-
"""train_and_run_policy.py

Train a policy using behavioral cloning (BC) and run it in a custom environment.
"""

import os
import h5py
import imageio
import numpy as np
import torch
from copy import deepcopy

# Robomimic imports
import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import BC
from robomimic.config import config_factory
import robosuite
from robosuite.environments.base import REGISTERED_ENVS
from robot_learning.color_sorting_env import ColorSortingEnv

# --------------------------
# Define paths
# --------------------------
DATASET_PATH = "../data/low_dim_v141.hdf5"  # dataset path
CHECKPOINT_PATH = "trained_policy.pth"
OUTPUT_FOLDER = "results"
VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "rollout_results.mp4")

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# Load Dataset
# --------------------------
assert os.path.exists(DATASET_PATH), "Dataset not found!"
dataset = h5py.File(DATASET_PATH, "r")
print(f"Loaded dataset with keys: {list(dataset.keys())}")

# Extract available demonstrations
demo_keys = list(dataset["data"].keys())
if not demo_keys:
    raise KeyError("No demonstrations found in dataset!")

# Use first demo to determine observation space and action dimensions
first_demo = demo_keys[0]

# Extract first available observation key
obs_keys = list(dataset[f"data/{first_demo}/obs"].keys())
if not obs_keys:
    raise KeyError(f"No observations found in {first_demo}!")

first_obs_key = obs_keys[0]

# Corrected obs_space extraction
obs_key_shapes = {first_obs_key: dataset[f"data/{first_demo}/obs/{first_obs_key}"].shape[1:]}
action_dim = dataset[f"data/{first_demo}/actions"].shape[-1]

print(f"Using '{first_demo}' for obs_key_shapes: {obs_key_shapes}, action_dim: {action_dim}")

# --------------------------
# Register Environment
# --------------------------
REGISTERED_ENVS["ColorSortingEnv"] = ColorSortingEnv
env_metadata = {
    "env_args": {
        "env_name": "ColorSortingEnv",
        "type": 1,
        "env_kwargs": {
            "robots": ["UR5e"],
            "has_renderer": True,
            "has_offscreen_renderer": True,
            "render_camera": "frontview",
            "ignore_done": False,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 5,
        }
    }
}

env = EnvUtils.create_env_from_metadata(env_metadata["env_args"], render=True, render_offscreen=True)

# --------------------------
# Configure Policy
# --------------------------
config = config_factory(algo_name="bc")  # Behavioral Cloning
config.unlock()
config.train.num_epochs = 100  # Training epochs
config.train.batch_size = 32

# Handle different Robomimic versions
if hasattr(config.algo, "policy"):
    config.algo.policy.hidden_dim = 256
    config.algo.policy.rnn.enabled = False
elif hasattr(config.algo, "network"):
    config.algo.network.hidden_dim = 256
    config.algo.network.rnn.enabled = False
else:
    raise ValueError("Unable to find 'actor', 'policy', or 'network' in config.algo")

config.lock()

# --------------------------
# Initialize BC Model
# --------------------------
# Initialize ObsUtils before BC model creation
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_key_shapes})

device = torch.device("cpu")  # Use CPU since no GPU is available

policy = BC(
    algo_config=config.algo,
    obs_config=config.observation,
    global_config=config,
    obs_key_shapes=obs_key_shapes,
    ac_dim=action_dim,
    device=device
)

# Training Loop (Only Using First Demonstration)
first_demo = list(dataset["data"].keys())[0]  # Select the first demo

print(f"Training policy on the first demonstration: {first_demo}")

# Extract observations and actions from the first demo
obs = dataset[f"data/{first_demo}/obs/{first_obs_key}"][:]  # Extract first observation key
actions = dataset[f"data/{first_demo}/actions"][:]  # Extract actions

num_epochs = config.train.num_epochs
batch_size = config.train.batch_size

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for i in range(0, len(obs), batch_size):
        obs_batch = obs[i : i + batch_size]
        action_batch = actions[i : i + batch_size]

        # Convert to tensors
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=device)

        # Ensure correct batch format
        batch = {"obs": {first_obs_key: obs_batch}, "actions": action_batch}

        # Debug batch structure
        print("Batch structure before training step:", batch.keys())
        print(f"Observation keys: {batch['obs'].keys()}")
        print(f"Observation shape: {batch['obs'][first_obs_key].shape}")
        print(f"Action shape: {batch['actions'].shape}")

        try:
            loss_dict = policy._train_step(batch)
            print(f"Loss dictionary: {loss_dict}")
        except Exception as e:
            print("Error during training step:", e)
            raise  # Re-raise to see the exact issue

        if not isinstance(loss_dict, dict):
            raise ValueError(f"Expected loss_dict to be a dictionary, but got {type(loss_dict)} instead.")

        # Debugging: Print available keys
        print(f"Available loss_dict keys: {loss_dict.keys()}")

        # Use the correct loss key
        if "loss" in loss_dict:
            loss = loss_dict["loss"]
        elif "action_loss" in loss_dict:
            loss = loss_dict["action_loss"]
        else:
            # If neither "loss" nor "action_loss" is found, use the first available key
            loss_keys = list(loss_dict.keys())
            if loss_keys:
                loss = loss_dict[loss_keys[0]]
                print(f"Warning: 'loss' not found. Using '{loss_keys[0]}' instead.")
            else:
                raise KeyError("No valid loss key found in loss_dict.")

        print(f"Loss at Epoch {epoch+1}, Batch {i//batch_size + 1}: {loss}")
        

# Save trained policy
policy.save(CHECKPOINT_PATH)
print(f"Trained policy saved to {CHECKPOINT_PATH}")


# --------------------------
# Load Trained Policy
# --------------------------
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=CHECKPOINT_PATH, device=device, verbose=True)

# --------------------------
# Rollout Function to Evaluate Policy
# --------------------------
def rollout(policy, env, horizon=400, render=False, video_writer=None, video_skip=5, camera_names=None):
    policy.start_episode()
    obs = env.reset_to(env.get_state())
    total_reward = 0.
    success = False

    for step_i in range(horizon):
        act = policy(ob=obs)
        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = env.is_success()["task"]

        if render:
            env.render(mode="human", camera_name=camera_names[0])

        if video_writer and step_i % video_skip == 0:
            frames = [
                env.render(mode="rgb_array", height=512, width=512, camera_name=cam)
                for cam in camera_names
            ]
            video_writer.append_data(np.concatenate(frames, axis=1))

        if done or success:
            break
        obs = deepcopy(next_obs)

    return {"Return": total_reward, "Horizon": step_i + 1, "Success_Rate": float(success)}

# --------------------------
# Run Policy and Save Video
# --------------------------
video_writer = imageio.get_writer(VIDEO_PATH, fps=20)

stats = rollout(
    policy=policy,
    env=env,
    horizon=400,
    render=False,
    video_writer=video_writer,
    video_skip=5,
    camera_names=["agentview"]
)

video_writer.close()
print(f"Rollout video saved to {VIDEO_PATH}")
print("Rollout statistics:", stats)
