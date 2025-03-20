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
import torch.nn.functional as F  # Import padding functions
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
import matplotlib.pyplot as plt  # Import for plotting

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

env = EnvUtils.create_env_from_metadata(env_metadata["env_args"], render=True, render_offscreen=True)

# --------------------------
# Configure Policy
# --------------------------
config = config_factory(algo_name="bc")  # Behavioral Cloning
config.unlock()
config.train.num_epochs = 100  # Training epochs
config.train.batch_size = 2
config.train.learning_rate = 1e-4  # Experiment with a higher value

# Handle different Robomimic versions
if hasattr(config.algo, "policy"):
    config.algo.policy.hidden_dim = 256
    config.algo.policy.rnn.enabled = True
elif hasattr(config.algo, "network"):
    config.algo.network.hidden_dim = 256
    config.algo.network.rnn.enabled = True
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

# Store losses and metrics
loss_history = []
policy_grad_norms_history = []

# Select first demonstration
first_demo = list(dataset["data"].keys())[0]

# Extract observations and actions from the first demo
obs = dataset[f"data/{first_demo}/obs/{first_obs_key}"][:]  
actions = dataset[f"data/{first_demo}/actions"][:]  

num_epochs = config.train.num_epochs
batch_size = config.train.batch_size

# Check initial loss before training
print("\nChecking Initial Loss Before Training...")
initial_batch = {
    "obs": {first_obs_key: torch.tensor(obs[:batch_size], dtype=torch.float32, device=device)},
    "goal_obs": {first_obs_key: torch.tensor(dataset[f"data/{first_demo}/next_obs/{first_obs_key}"][:batch_size], dtype=torch.float32, device=device)},
    "actions": torch.tensor(actions[:batch_size], dtype=torch.float32, device=device)
}
predictions = policy._forward_training(initial_batch)
initial_losses = policy._compute_losses(predictions, initial_batch)
print(f"Initial Loss: {initial_losses}")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    epoch_policy_grad_norm = 0
    num_batches = 0  # Count batches to average the loss per epoch

    for i in range(0, len(obs), batch_size):
        obs_batch = obs[i : i + batch_size]
        action_batch = actions[i : i + batch_size]
        next_obs_batch = dataset[f"data/{first_demo}/next_obs/{first_obs_key}"][i : i + batch_size]

        # Convert to tensors
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)

        # Ensure correct batch format
        batch = {
            "obs": {first_obs_key: obs_batch},
            "goal_obs": {first_obs_key: next_obs_batch},  # Use next_obs instead
            "actions": action_batch
        }

        # Forward pass to get predictions
        predictions = policy._forward_training(batch)

        # Compute loss using predictions and batch
        losses = policy._compute_losses(predictions, batch)

        # Ensure action loss exists
        if "action_loss" in losses:
            loss_value = losses["action_loss"].item()
        elif "loss" in losses:
            loss_value = losses["loss"].item()
        else:
            loss_value = 0  # Default to 0 if no valid loss key is found

        # Perform gradient update
        loss_dict = policy._train_step(losses)

        # Store loss values
        epoch_loss += loss_value
        epoch_policy_grad_norm += loss_dict["policy_grad_norms"]
        num_batches += 1

        print(f"Batch {i//batch_size + 1}: Loss={loss_value}, Grad Norm={loss_dict['policy_grad_norms']}")

        # Debug: Check if gradients are updating
        for name, param in policy.nets["policy"].named_parameters():
            if param.grad is not None:
                print(f"Gradient Norm for {name}: {param.grad.norm().item()}")

    # Average loss and gradient norm per epoch
    loss_history.append(epoch_loss / max(1, num_batches))
    policy_grad_norms_history.append(epoch_policy_grad_norm / max(1, num_batches))

    print(f"Epoch {epoch+1} Completed: Avg Loss={loss_history[-1]}, Avg Grad Norm={policy_grad_norms_history[-1]}")



# --------------------------
# Plot Training Metrics
# --------------------------

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Action Loss", marker='o')
plt.plot(policy_grad_norms_history, label="Policy Grad Norms", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss / Gradient Norms")
plt.title("Training Loss and Policy Gradient Norms")
plt.legend()
plt.grid()

# Define save path
loss_plot_path = os.path.join(OUTPUT_FOLDER, "training_loss_plot.png")

# Save the figure
plt.savefig(loss_plot_path, dpi=300)  # Save as high-quality PNG
plt.close()  # Close the figure to prevent displaying

print(f"Training loss plot saved to {loss_plot_path}")

# --------------------------
# Rollout Function to Evaluate Policy
# --------------------------
# --------------------------
# Rollout Function to Evaluate Policy
# --------------------------

def rollout(policy, env, horizon=400, render=False, video_writer=None, video_skip=5, camera_names=None):
    obs = env.reset()  # Reset environment
    total_reward = 0.
    success = False

    for step_i in range(horizon):
        # Extract observation keys
        obs_keys = list(obs.keys())
        #print(f"Available observation keys: {obs_keys}")  # Debugging

        # Ensure the observation dictionary has the expected input format
        obs_tensor = torch.tensor(obs["object"], dtype=torch.float32).unsqueeze(0)  # Shape: [1, 8]

        # Expected input shape
        expected_shape = 10  # Policy expects [1, 10]

        # Pad if necessary
        if obs_tensor.shape[-1] < expected_shape:
            pad_size = expected_shape - obs_tensor.shape[-1]
            obs_tensor = F.pad(obs_tensor, (0, pad_size), "constant", 0)  # Zero-pad
            #print(f"Padded observation to shape: {obs_tensor.shape}")  # Debugging

        # Create observation dictionary
        obs_dict = {"object": obs_tensor}  # Now Shape: [1, 10]

        # Get action from policy
        with torch.no_grad():
            policy_output = policy.nets["policy"](obs_dict)  # Run policy forward pass
            
            if isinstance(policy_output, dict):
                action = policy_output["actions"]  # Get actions if output is a dict
            else:
                action = policy_output  # Assume it's already the action tensor
            
            #print(f"Raw action shape: {action.shape}")  # Debugging
            action = action.cpu().numpy().squeeze()

        # Step the environment
        next_obs, r, done, _ = env.step(action)
        total_reward += r
        success = env.is_success()["task"]

        if render:
            env.render(mode="rgb_array", height=912, width=1520, camera_name="frontview")

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
    camera_names=["frontview"]
)

video_writer.close()
print(f"Rollout video saved to {VIDEO_PATH}")
print("Rollout statistics:", stats)
