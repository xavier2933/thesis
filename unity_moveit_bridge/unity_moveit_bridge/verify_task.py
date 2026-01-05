from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

# Load your dataset
repo_id = "Xavier033/pick_place_LIBERO"
dataset = LeRobotDataset(repo_id)

# In the class you shared, metadata is in dataset.meta
meta = dataset.meta

print(f"--- Dataset Summary: {repo_id} ---")
print(f"Total Episodes: {meta.total_episodes}")
print(f"Total Frames:   {meta.total_frames}")
print("-" * 40)

for ep_idx in range(meta.total_episodes):
    # Access the episode dictionary using the index
    ep_info = meta.episodes[ep_idx]
    
    # Use the keys found in your _save_episode_metadata function
    start_idx = ep_info["dataset_from_index"]
    end_idx = ep_info["dataset_to_index"]
    length = end_idx - start_idx
    
    # Grab the frame data from the main dataset
    frame = dataset[start_idx]
    
    # Task verification
    task_value = frame.get("task", "MISSING")
    
    # Handle if task is a list/tensor or a raw string
    if isinstance(task_value, (list, np.ndarray)) and len(task_value) > 0:
        task_string = task_value[0]
    else:
        task_string = task_value

    if task_string in ["", None, "MISSING"]:
        status = "❌ FAILED (Empty/Missing)"
    else:
        status = f"✅ PASSED: '{task_string}'"
    
    print(f"Episode {ep_idx}: {status} ({length} frames, Indices {start_idx}-{end_idx})")

# Verify shapes
first_frame = dataset[0]
print("-" * 40)
print(f"State Vector Shape: {first_frame['observation.state'].shape} (Expected: [8])")
print(f"Action Vector Shape: {first_frame['action'].shape} (Expected: [7])")