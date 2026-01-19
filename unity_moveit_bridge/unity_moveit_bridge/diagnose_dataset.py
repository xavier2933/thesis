#!/usr/bin/env python3
"""
Dataset Diagnostics Script
Analyzes recorded data for issues:
- Gripper signal flipping
- Action magnitude validation
- State-action consistency
"""
import numpy as np
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def analyze_episode(dataset, episode_idx):
    """Analyze a single episode for issues"""
    print(f"\n{'='*60}")
    print(f"ANALYZING EPISODE {episode_idx}")
    print(f"{'='*60}\n")
    
    # Get frame indices
    if hasattr(dataset, 'episode_data_index'):
        ep_meta = dataset.episode_data_index
        start_idx = ep_meta['from'][episode_idx].item()
        end_idx = ep_meta['to'][episode_idx].item()
    else:
        indices = [i for i in range(len(dataset)) if dataset[i]['episode_index'] == episode_idx]
        start_idx = min(indices)
        end_idx = max(indices) + 1
    
    num_frames = end_idx - start_idx
    print(f"📊 Episode has {num_frames} frames\n")
    
    # Collect data
    actions = []
    states = []
    gripper_vals = []
    
    for i in range(start_idx, end_idx):
        frame = dataset[i]
        
        action = frame['action']
        state = frame['observation.state']
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        actions.append(action)
        states.append(state)
        gripper_vals.append(action[6])  # Gripper from action
    
    actions = np.array(actions)
    states = np.array(states)
    gripper_vals = np.array(gripper_vals)
    
    # === 1. GRIPPER ANALYSIS ===
    print("🤖 GRIPPER ANALYSIS:")
    print("-" * 40)
    
    # Check for flips
    gripper_changes = np.diff(gripper_vals)
    large_changes = np.where(np.abs(gripper_changes) > 1.5)[0]
    
    print(f"Gripper value range: [{gripper_vals.min():.2f}, {gripper_vals.max():.2f}]")
    print(f"Expected: -1.0 (closed) or 1.0 (open)")
    print(f"Number of gripper state changes: {len(large_changes)}")
    
    if len(large_changes) > 0:
        print(f"\n⚠️ Gripper changes detected at frames:")
        for idx in large_changes:
            print(f"  Frame {idx}: {gripper_vals[idx]:.2f} -> {gripper_vals[idx+1]:.2f}")
    
    # Check gripper consistency with state
    state_gripper = states[:, 6]  # grip_l from state
    print(f"\nState gripper range: [{state_gripper.min():.2f}, {state_gripper.max():.2f}]")
    
    # Check if action gripper matches state gripper
    gripper_mismatch = np.where(np.abs(gripper_vals - state_gripper) > 0.1)[0]
    if len(gripper_mismatch) > 0:
        print(f"⚠️ WARNING: {len(gripper_mismatch)} frames have action/state gripper mismatch!")
        print(f"  First few mismatches at frames: {gripper_mismatch[:5]}")
    
    # === 2. ACTION MAGNITUDE ANALYSIS ===
    print(f"\n📏 ACTION MAGNITUDE ANALYSIS:")
    print("-" * 40)
    
    position_deltas = actions[:, :3]
    rotation_deltas = actions[:, 3:6]
    
    pos_mags = np.linalg.norm(position_deltas, axis=1)
    rot_mags = np.linalg.norm(rotation_deltas, axis=1)
    
    print(f"Position deltas (meters):")
    print(f"  Mean: {pos_mags.mean():.5f}")
    print(f"  Std:  {pos_mags.std():.5f}")
    print(f"  Max:  {pos_mags.max():.5f}")
    print(f"  Min:  {pos_mags.min():.5f}")
    
    print(f"\nRotation deltas (radians):")
    print(f"  Mean: {rot_mags.mean():.5f}")
    print(f"  Std:  {rot_mags.std():.5f}")
    print(f"  Max:  {rot_mags.max():.5f}")
    
    # Check for suspiciously large deltas (at 10Hz, should be small)
    large_pos_deltas = np.where(pos_mags > 0.05)[0]  # More than 5cm
    if len(large_pos_deltas) > 0:
        print(f"\n⚠️ WARNING: {len(large_pos_deltas)} frames have large position deltas (>5cm)")
        print(f"  Frames: {large_pos_deltas[:10]}")  # Show first 10
        print(f"  This suggests actions may be 'delta-to-goal' not 'delta-per-step'")
    
    # Check for zero deltas (robot not moving)
    zero_deltas = np.where(pos_mags < 0.0001)[0]
    if len(zero_deltas) > 0:
        print(f"\n⚠️ INFO: {len(zero_deltas)} frames have near-zero deltas")
        if zero_deltas[0] == 0:
            print(f"  First frame is zero (expected if using consecutive deltas)")
    
    # === 3. STATE RECONSTRUCTION TEST ===
    print(f"\n🔬 STATE RECONSTRUCTION TEST:")
    print("-" * 40)
    print("Integrating actions to reconstruct trajectory...")
    
    # Start from first state
    integrated_pos = states[0, :3].copy()
    integrated_rpy = states[0, 3:6].copy()
    
    max_error = 0
    errors = []
    
    for i in range(1, len(actions)):
        # Integrate action
        integrated_pos += actions[i, :3]
        integrated_rpy += actions[i, 3:6]
        
        # Compare to recorded state
        recorded_pos = states[i, :3]
        error = np.linalg.norm(integrated_pos - recorded_pos)
        errors.append(error)
        max_error = max(max_error, error)
    
    errors = np.array(errors)
    
    print(f"Position reconstruction error (meters):")
    print(f"  Mean: {errors.mean():.5f}")
    print(f"  Max:  {errors.max():.5f}")
    print(f"  Final: {errors[-1]:.5f}")
    
    if errors[-1] < 0.01:
        print(f"✅ PASS: Actions correctly represent consecutive state deltas")
    elif errors[-1] < 0.05:
        print(f"⚠️ WARNING: Moderate drift - may indicate accumulation error")
    else:
        print(f"❌ FAIL: Large error - actions likely don't match state changes")
    
    # === 4. VISUALIZATION ===
    print(f"\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Position deltas over time
    ax = axes[0, 0]
    ax.plot(pos_mags, label='Position delta magnitude')
    ax.axhline(y=0.05, color='r', linestyle='--', label='5cm threshold')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Magnitude (m)')
    ax.set_title('Position Delta Magnitudes')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Gripper signal
    ax = axes[0, 1]
    ax.plot(gripper_vals, label='Action gripper')
    ax.plot(state_gripper, label='State gripper', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Gripper value')
    ax.set_title('Gripper Signal (1=open, -1=closed)')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Reconstruction error
    ax = axes[1, 0]
    ax.plot(errors)
    ax.axhline(y=0.01, color='g', linestyle='--', label='1cm (good)')
    ax.axhline(y=0.05, color='orange', linestyle='--', label='5cm (acceptable)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (m)')
    ax.set_title('State Reconstruction Error')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: XYZ trajectory
    ax = axes[1, 1]
    ax.plot(states[:, 0], states[:, 1], 'b-', label='XY trajectory')
    ax.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start', zorder=5)
    ax.scatter(states[-1, 0], states[-1, 1], c='r', s=100, label='End', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('End-Effector XY Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'episode_{episode_idx}_diagnostics.png', dpi=150)
    print(f"✅ Saved plot to: episode_{episode_idx}_diagnostics.png")
    
    return {
        'num_frames': num_frames,
        'gripper_changes': len(large_changes),
        'large_deltas': len(large_pos_deltas),
        'final_error': errors[-1],
        'mean_pos_delta': pos_mags.mean(),
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose dataset issues")
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0, help="Episode to analyze")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Analyze all episodes")
    
    args = parser.parse_args()
    
    print(f"📂 Loading dataset: {args.repo}")
    dataset = LeRobotDataset(args.repo, root=args.root)
    print(f"✅ Loaded {dataset.num_episodes} episodes")
    
    if args.all:
        print("\n🔍 Analyzing ALL episodes...")
        results = []
        for ep in range(dataset.num_episodes):
            result = analyze_episode(dataset, ep)
            results.append(result)
        
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL EPISODES:")
        print(f"{'='*60}")
        print(f"Total episodes: {len(results)}")
        print(f"Average frames per episode: {np.mean([r['num_frames'] for r in results]):.1f}")
        print(f"Episodes with gripper changes: {sum(1 for r in results if r['gripper_changes'] > 0)}")
        print(f"Episodes with large deltas: {sum(1 for r in results if r['large_deltas'] > 0)}")
        print(f"Mean final reconstruction error: {np.mean([r['final_error'] for r in results]):.5f}m")
    else:
        analyze_episode(dataset, args.episode)

if __name__ == "__main__":
    main()