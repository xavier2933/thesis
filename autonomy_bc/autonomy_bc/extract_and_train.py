#!/usr/bin/env python3
"""
Extract demonstrations from ROS2 bag and train behavior cloning model
"""
import numpy as np
import sqlite3
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ROS2 message parsing
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

class TrajectoryExtractor:
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)
        self.demonstrations = []
        
    def extract_from_bag(self):
        """Extract state-action pairs from ROS2 bag"""
        print(f"Extracting from bag: {self.bag_path}")
        
        # ROS2 bags are SQLite databases
        db_path = self.bag_path / "demo_session2_0.db3"  # Adjust filename
        if not db_path.exists():
            # Find the actual db3 file
            db_files = list(self.bag_path.glob("*.db3"))
            if not db_files:
                raise FileNotFoundError(f"No .db3 files found in {self.bag_path}")
            db_path = db_files[0]
            
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get topic info
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        unity_poses = []
        joint_states = []
        
        # Extract unity target poses
        if '/unity_target_pose' in topics:
            topic_id = topics['/unity_target_pose']
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (topic_id,))
            
            pose_msg_type = get_message('geometry_msgs/msg/Pose')
            for timestamp, data in cursor.fetchall():
                msg = deserialize_message(data, pose_msg_type)
                unity_poses.append({
                    'timestamp': timestamp,
                    'position': [msg.position.x, msg.position.y, msg.position.z],
                    'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                })
        
        # Extract joint states  
        if '/joint_states' in topics:
            topic_id = topics['/joint_states']
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (topic_id,))
            
            joint_msg_type = get_message('sensor_msgs/msg/JointState')
            for timestamp, data in cursor.fetchall():
                msg = deserialize_message(data, joint_msg_type)
                # Only take first 7 joints (Panda arm)
                joint_states.append({
                    'timestamp': timestamp,
                    'positions': list(msg.position[:7])
                })
        
        conn.close()
        
        print(f"Extracted {len(unity_poses)} unity poses and {len(joint_states)} joint states")
        return self._create_trajectories(unity_poses, joint_states)
    
    def _create_trajectories(self, unity_poses, joint_states):
        """Create state-action pairs from extracted data"""
        trajectories = []
        
        # Sort by timestamp
        unity_poses.sort(key=lambda x: x['timestamp'])
        joint_states.sort(key=lambda x: x['timestamp'])
        
        # Match poses to joint states by timestamp (within 50ms)
        matched_pairs = []
        tolerance = 50_000_000  # 50ms in nanoseconds
        
        for pose in unity_poses:
            # Find closest joint state
            closest_joint = None
            min_diff = float('inf')
            
            for joint in joint_states:
                diff = abs(joint['timestamp'] - pose['timestamp'])
                if diff < min_diff and diff < tolerance:
                    min_diff = diff
                    closest_joint = joint
            
            if closest_joint:
                matched_pairs.append({
                    'current_joints': closest_joint['positions'],
                    'target_pose': pose['position'] + pose['orientation']  # [x,y,z,qx,qy,qz,qw]
                })
        
        # Split into episodes based on reset events (large jumps in target pose)
        episodes = []
        current_episode = []
        
        for i, pair in enumerate(matched_pairs):
            if i > 0:
                # Check for large jump (reset event)
                prev_pos = np.array(matched_pairs[i-1]['target_pose'][:3])
                curr_pos = np.array(pair['target_pose'][:3])
                
                if np.linalg.norm(curr_pos - prev_pos) > 0.3:  # 30cm jump = reset
                    if len(current_episode) > 5:  # Only save episodes with >5 steps
                        episodes.append(current_episode)
                    current_episode = []
            
            current_episode.append(pair)
        
        # Add final episode
        if len(current_episode) > 5:
            episodes.append(current_episode)
        
        print(f"Created {len(episodes)} episodes")
        return episodes

class BehaviorCloningModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=7, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Limit output range
        )
    
    def forward(self, x):
        return self.network(x)

def train_behavior_cloning(states, actions, episodes, model_save_path="robot_bc_model.pth"):
    """Train behavior cloning model"""

    print(f"Training data: {states.shape[0]} samples")
    print(f"State shape: {states.shape[1]}, Action shape: {actions.shape[1]}")
    
    # Normalize data
    state_scaler = StandardScaler()
    action_scaler = StandardScaler()
    
    states_normalized = state_scaler.fit_transform(states)
    actions_normalized = action_scaler.fit_transform(actions)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        states_normalized, actions_normalized, test_size=0.2, random_state=42
    )
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = BehaviorCloningModel(
        state_dim=states.shape[1], 
        action_dim=actions.shape[1]
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    for epoch in range(200):
        # Training
        model.train()
        train_loss = 0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions in test_loader:
                predictions = model(batch_states)
                loss = criterion(predictions, batch_actions)
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    # Save model and scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_scaler': state_scaler,
        'action_scaler': action_scaler,
        'model_config': {
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1]
        }
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Behavior Cloning Training')
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, state_scaler, action_scaler

def load_all_data(episodes_from_bag, dagger_data_dir='../data'):
    """Combine original bag episodes with DAgger sessions"""
    from pathlib import Path
    import glob
    
    # Start with original episodes
    all_states = []
    all_actions = []
    
    # Convert bag episodes to state-action pairs
    for episode in episodes_from_bag:
        for step in episode:
            all_states.append(step['current_joints'])
            all_actions.append(step['target_pose'])
    
    original_count = len(all_states)
    
    # Load all DAgger sessions
    dagger_dir = Path(dagger_data_dir)
    dagger_files = sorted(dagger_dir.glob('dagger_session_*.pkl'))
    
    if dagger_files:
        print(f"\nFound {len(dagger_files)} DAgger session files")
        
        for dagger_file in dagger_files:
            with open(dagger_file, 'rb') as f:
                dagger_data = pickle.load(f)
            
            print(f"  Loading {dagger_file.name}: {len(dagger_data)} samples")
            
            for sample in dagger_data:
                all_states.append(sample['state'])
                all_actions.append(sample['action'])
        
        dagger_count = len(all_states) - original_count
        print(f"\nTotal samples: {len(all_states)}")
        print(f"  - Original (from bag): {original_count}")
        print(f"  - DAgger interventions: {dagger_count}")
        print(f"  - DAgger percentage: {100 * dagger_count / len(all_states):.1f}%")
    else:
        print("\nNo DAgger sessions found - training on bag data only")
    
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)

def main():
    print("=== BEHAVIOR CLONING PIPELINE ===")
    
    # Extract demonstrations
    bag_path = "demo_session2"  # Your bag directory
    print(f"1. Extracting demonstrations from: {bag_path}")
    extractor = TrajectoryExtractor(bag_path)
    episodes = extractor.extract_from_bag()

    states, actions = load_all_data(
        episodes_from_bag=episodes,
        dagger_data_dir='../data'  # Adjust path as needed
    )

    train_behavior_cloning(
        states=states,
        actions=actions,
        model_save_path="robot_bc_model.pth"
    )

if __name__ == "__main__":
    main()
