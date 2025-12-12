#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def validate(file_path='dataset.parquet'):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    print(f"Loaded {len(df)} frames.")
    print("Columns:", df.columns.tolist())

    if len(df) == 0:
        print("Dataset is empty.")
        return

    # Check for list columns and stack them
    # observation.state expected to be 8 dims (7 joints + 1 gripper)
    # action expected to be 5 dims (dx, dy, dz, dwrist, dgripper)
    
    timestamps = df['timestamp'].values
    
    # Expand observation.state
    # Assuming it's stored as a list/array in the parquet cell
    try:
        obs_states = np.stack(df['observation.state'].values)
        actions = np.stack(df['action'].values)
    except Exception as e:
        print(f"Error flattening arrays (shapes might be inconsistent): {e}")
        # Fallback for debugging
        print("First row observation:", df['observation.state'].iloc[0])
        return

    print(f"Observation shape: {obs_states.shape}")
    print(f"Action shape: {actions.shape}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # 1. Joint States (First 7)
    for i in range(min(7, obs_states.shape[1])):
        axes[0].plot(timestamps, obs_states[:, i], label=f'Joint {i+1}')
    axes[0].set_title('Joint States (Positions)')
    axes[0].set_ylabel('Rad')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    axes[0].grid(True)

    # 2. Gripper State (8th element)
    if obs_states.shape[1] >= 8:
        axes[1].plot(timestamps, obs_states[:, 7], label='Gripper State', color='orange')
        axes[1].set_title('Gripper State')
        axes[1].set_ylabel('Width/State')
        axes[1].legend()
        axes[1].grid(True)

    # 3. Actions (Deltas)
    action_labels = ['dx', 'dy', 'dz', 'd_wrist', 'd_gripper']
    for i in range(min(5, actions.shape[1])):
        axes[2].plot(timestamps, actions[:, i], label=action_labels[i] if i < 5 else f'Act {i}')
    axes[2].set_title('Actions (Deltas)')
    axes[2].set_ylabel('Delta')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    axes[2].grid(True)

    plt.tight_layout()
    output_file = 'validation_plots.png'
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    validate()
