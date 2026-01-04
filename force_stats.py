from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "Xavier033/test_pick_place"

# This will download the dataset and attempt to parse the meta/info.json
try:
    dataset = LeRobotDataset(repo_id)
    print(f"✅ Success! Dataset loaded.")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Features: {list(dataset.features.keys())}")
    
    # Try fetching the first frame
    sample = dataset[0]
    print("✅ Successfully accessed first frame.")
except Exception as e:
    print(f"❌ Format Error: {e}")