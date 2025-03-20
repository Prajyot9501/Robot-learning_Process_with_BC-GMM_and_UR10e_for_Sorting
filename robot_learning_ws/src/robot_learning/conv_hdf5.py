import json
import h5py
import numpy as np

"""
Expected JSON structure (pick_and_place_log_5.json):
{
  "Timestep 0": {
      "Color": "yellow",
      "Operation": "Default",
      "Observations": {
          "object": [0.0, 0.0, ...], 
          "robot0_eef_pos": [...],
          "robot0_eef_quat": [...],
          ...
      },
      "Action": [...]
  },
  "Timestep 1": {...},
  ...
}
"""

def create_empty_dataset(group, name):
    """
    Helper to create an empty dataset if needed
    (though you might not actually need to store truly empty datasets).
    """
    group.create_dataset(name, data=[])

def create_obs_group(parent_group, obs_dict):
    """
    Create an 'obs' or 'next_obs' subgroup and fill it with individual datasets
    like object, robot0_eef_pos, ...
    """
    for key, val in obs_dict.items():
        val_array = np.array(val, dtype=np.float32)
        parent_group.create_dataset(key, data=val_array)

def json_to_hdf5(json_path, hdf5_path):
    with open(json_path, 'r') as f:
        data_json = json.load(f)

    with h5py.File(hdf5_path, 'w') as hf:
        # 1) Create the top-level "data" group and an empty "mask" group
        data_group = hf.create_group("data")
        hf.create_group("mask")  # empty group

        # 2) Sort timesteps by key (optional) if you want them in order: Timestep 0, 1, 2...
        timestep_keys = sorted(data_json.keys(), key=lambda x: int(x.split()[-1]))

        # We'll treat each Timestep as its own 'demo_i'
        for i, ts_key in enumerate(timestep_keys):
            # e.g., "demo_0", "demo_1", etc.
            demo_name = f"demo_{i}"
            demo_group = data_group.create_group(demo_name)

            ts_data = data_json[ts_key]  # e.g. { "Color":..., "Observations":..., "Action":... }

            # 3) Fill "obs" group from the Timestep's Observations
            obs_group = demo_group.create_group("obs")
            observations = ts_data.get("Observations", {})
            create_obs_group(obs_group, observations)

            # 4) Create next_obs group
            #    Typically, you'd want the next timestep's Observations as "next_obs" if available.
            if i < len(timestep_keys) - 1:
                # let's look ahead to Timestep i+1
                next_ts_key = timestep_keys[i + 1]
                next_obs = data_json[next_ts_key].get("Observations", {})
            else:
                next_obs = {}
            next_obs_group = demo_group.create_group("next_obs")
            create_obs_group(next_obs_group, next_obs)

            # 5) actions dataset
            #    We assume Action is a list of floats
            actions = ts_data.get("Action", [])
            actions_array = np.array(actions, dtype=np.float32)
            demo_group.create_dataset("actions", data=actions_array)

            # 6) dones dataset
            #    You may want it to be 1.0 if last step, else 0.0, or some logic:
            done_val = [1.0 if i == len(timestep_keys) - 1 else 0.0]
            demo_group.create_dataset("dones", data=np.array(done_val, dtype=np.float32))

            # 7) rewards dataset
            #    Fill with your logic. We might store a 0.0 for every step except last step = 1.0, or all zeros, etc.
            reward_val = [1.0 if i == len(timestep_keys) - 1 else 0.0]
            demo_group.create_dataset("rewards", data=np.array(reward_val, dtype=np.float32))

            # 8) states dataset
            #    If you have an internal notion of 'states'. Otherwise, store an empty dataset or array of zeros
            #    For demonstration, let's store an empty dataset
            create_empty_dataset(demo_group, "states")

    print(f"Created HDF5 file: {hdf5_path}")


if __name__ == "__main__":
    json_file = "data/pick_and_place_log_5.json"
    out_hdf5 = "data/pick_and_place_5.hdf5"
    json_to_hdf5(json_file, out_hdf5)
