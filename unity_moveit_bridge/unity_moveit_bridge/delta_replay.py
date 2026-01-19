#!/usr/bin/env python3
"""
Delta Replay Script - Validates Action Recording
Reconstructs trajectory by integrating action deltas from the initial state.
This verifies that your recorded deltas can actually reproduce the motion.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import argparse
import time
import threading
import torch
from pathlib import Path

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DeltaReplayNode(Node):
    def __init__(self, repo_id, root_dir, episode_idx, fps=10, exec_duration=0.8):
        super().__init__("delta_replay_node")
        
        self.repo_id = repo_id
        self.root_dir = root_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.exec_duration = exec_duration
        
        # Load Dataset
        self.get_logger().info(f"📂 Loading dataset {repo_id}...")
        self.dataset = LeRobotDataset(repo_id, root=root_dir)
        self.get_logger().info(f"✅ Dataset loaded. Found {self.dataset.num_episodes} episodes.")
        
        if self.episode_idx >= self.dataset.num_episodes:
            self.get_logger().error(f"❌ Episode {self.episode_idx} out of range")
            raise ValueError("Episode index out of range")

        # Robot Configuration
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # Data buffers
        self.latest_eef_pose = None
        self.latest_joints = None
        self.data_lock = threading.Lock()
        
        # Publishers
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        self.gripper_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        
        # Subscribers
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        
        # IK Service
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        while not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("⏳ Waiting for IK service...")
            
    def eef_callback(self, msg):
        with self.data_lock:
            self.latest_eef_pose = msg
            
    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            with self.data_lock:
                self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    def reset_environment(self):
        self.get_logger().info("🔄 Resetting environment...")
        self.pub_aut.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_reset.publish(Bool(data=True))
        time.sleep(4.0)
        self.pub_aut.publish(Bool(data=True))
        time.sleep(1.0)
        self.get_logger().info("✅ Environment reset complete")

    def run_delta_replay(self):
        """Replay by integrating action deltas from initial state"""
        self.get_logger().info(f"▶️ Starting DELTA REPLAY of Episode {self.episode_idx}")
        self.get_logger().info("🔬 This validates that recorded deltas can reproduce the trajectory")
        
        # Get frame indices
        if hasattr(self.dataset, 'episode_data_index'):
            ep_meta = self.dataset.episode_data_index
            start_idx = ep_meta['from'][self.episode_idx].item()
            end_idx = ep_meta['to'][self.episode_idx].item()
        else:
            indices = [i for i in range(len(self.dataset)) if self.dataset[i]['episode_index'] == self.episode_idx]
            if not indices:
                self.get_logger().error("Could not find frames for this episode!")
                return
            start_idx = min(indices)
            end_idx = max(indices) + 1

        num_frames = end_idx - start_idx
        self.get_logger().info(f"🎞️ Frames: {start_idx} to {end_idx} ({num_frames} frames)")
        
        # Wait for initial state
        time.sleep(1.0)
        with self.data_lock:
            current_eef = self.latest_eef_pose
            current_joints = self.latest_joints
        
        if current_eef is None or current_joints is None:
            self.get_logger().error("❌ No robot state available!")
            return
        
        # Get initial recorded state from first frame
        first_frame = self.dataset[start_idx]
        initial_state = first_frame['observation.state']
        if isinstance(initial_state, torch.Tensor):
            initial_state = initial_state.cpu().numpy()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("INITIAL STATE COMPARISON:")
        self.get_logger().info(f"Recorded: XYZ=[{initial_state[0]:.3f}, {initial_state[1]:.3f}, {initial_state[2]:.3f}]")
        self.get_logger().info(f"Recorded: RPY=[{initial_state[3]:.3f}, {initial_state[4]:.3f}, {initial_state[5]:.3f}]")
        self.get_logger().info(f"Recorded: Grip=[{initial_state[6]:.3f}, {initial_state[7]:.3f}]")
        
        current_rpy = euler_from_quaternion([
            current_eef.orientation.x, current_eef.orientation.y,
            current_eef.orientation.z, current_eef.orientation.w
        ])
        self.get_logger().info(f"Current:  XYZ=[{current_eef.position.x:.3f}, {current_eef.position.y:.3f}, {current_eef.position.z:.3f}]")
        self.get_logger().info(f"Current:  RPY=[{current_rpy[0]:.3f}, {current_rpy[1]:.3f}, {current_rpy[2]:.3f}]")
        self.get_logger().info("=" * 60)
        
        # Initialize integrated state with CURRENT robot state (not recorded)
        # This tests if deltas work from ANY starting position
        integrated_pos = np.array([current_eef.position.x, current_eef.position.y, current_eef.position.z])
        integrated_rpy = np.array(current_rpy)
        
        # Track gripper state
        prev_gripper_val = None
        
        for frame_num, i in enumerate(range(start_idx, end_idx)):
            if not rclpy.ok():
                break
            
            frame = self.dataset[i]
            
            # Get action delta
            action = frame['action']
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
            
            # Get recorded state for comparison
            recorded_state = frame['observation.state']
            if isinstance(recorded_state, torch.Tensor):
                recorded_state = recorded_state.cpu().numpy()
            
            # Integrate delta into our reconstructed state
            integrated_pos += action[:3]  # Add position delta
            integrated_rpy += action[3:6]  # Add rotation delta
            gripper_val = action[6]
            
            # Log comparison every 10 frames
            if frame_num % 10 == 0:
                pos_error = np.linalg.norm(integrated_pos - recorded_state[:3])
                self.get_logger().info(
                    f"Frame {frame_num}/{num_frames} | "
                    f"Delta: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}] | "
                    f"Pos Error: {pos_error:.4f}m | "
                    f"Grip: {gripper_val:.2f}"
                )
            
            # Check for gripper signal issues
            if prev_gripper_val is not None:
                if abs(gripper_val - prev_gripper_val) > 1.5:  # Changed from -1 to 1 or vice versa
                    self.get_logger().warn(
                        f"⚠️ Gripper flip detected at frame {frame_num}: "
                        f"{prev_gripper_val:.2f} -> {gripper_val:.2f}"
                    )
            prev_gripper_val = gripper_val
            
            # Get current robot state for IK seeding
            with self.data_lock:
                current_joints = self.latest_joints
            
            # Build target pose from integrated state
            target_quat = quaternion_from_euler(*integrated_rpy)
            
            target_pose = Pose()
            target_pose.position.x = float(integrated_pos[0])
            target_pose.position.y = float(integrated_pos[1])
            target_pose.position.z = float(integrated_pos[2])
            target_pose.orientation.x = float(target_quat[0])
            target_pose.orientation.y = float(target_quat[1])
            target_pose.orientation.z = float(target_quat[2])
            target_pose.orientation.w = float(target_quat[3])
            
            # Control gripper
            gripper_open = bool(gripper_val > 0.0)
            self.gripper_pub.publish(Bool(data=gripper_open))
            
            # Publish for visualization
            self.target_pub.publish(target_pose)
            
            # Solve IK and move
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.base_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = target_pose
            
            joints = self.get_ik_solution(pose_stamped, current_joints)
            if joints:
                self.move_to_joints(joints, duration=self.exec_duration)
                time.sleep(self.exec_duration + 0.1)
            else:
                self.get_logger().warn(f"✗ IK failed for frame {frame_num}")
                # Continue anyway to see error accumulation
                time.sleep(0.1)
        
        # Final comparison
        self.get_logger().info("=" * 60)
        self.get_logger().info("FINAL STATE COMPARISON:")
        final_recorded = self.dataset[end_idx - 1]['observation.state']
        if isinstance(final_recorded, torch.Tensor):
            final_recorded = final_recorded.cpu().numpy()
        
        final_error = np.linalg.norm(integrated_pos - final_recorded[:3])
        self.get_logger().info(f"Integrated: XYZ=[{integrated_pos[0]:.3f}, {integrated_pos[1]:.3f}, {integrated_pos[2]:.3f}]")
        self.get_logger().info(f"Recorded:   XYZ=[{final_recorded[0]:.3f}, {final_recorded[1]:.3f}, {final_recorded[2]:.3f}]")
        self.get_logger().info(f"Position Error: {final_error:.4f}m")
        
        if final_error < 0.01:  # Less than 1cm error
            self.get_logger().info("✅ SUCCESS: Delta integration is accurate!")
        elif final_error < 0.05:  # Less than 5cm
            self.get_logger().warn("⚠️ WARNING: Moderate error - check for drift")
        else:
            self.get_logger().error("❌ FAILURE: Large error - deltas may be incorrect!")
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🏁 Delta Replay Complete")
            
    def get_ik_solution(self, pose_stamped, current_joints):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        # if current_joints is not None:
        #     req.ik_request.robot_state.joint_state.name = self.joint_names
        #     req.ik_request.robot_state.joint_state.position = current_joints.tolist()
        
        event = threading.Event()
        result_wrapper = {"res": None}
        def done_callback(future):
            result_wrapper["res"] = future.result()
            event.set()
        
        future = self.ik_client.call_async(req)
        future.add_done_callback(done_callback)
        
        if not event.wait(timeout=3.0):
            return None
        
        res = result_wrapper["res"]
        if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
            return list(res.solution.joint_state.position[:7])
        return None

    def move_to_joints(self, joints, duration=0.5):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = list(joints)
        point.velocities = [0.0] * 7
        point.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        traj.points = [point]
        self.arm_pub.publish(traj)

def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="Validate recorded deltas by integration")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID or local path")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--root", type=str, default=None, help="Root dir for dataset")
    parser.add_argument("--fps", type=int, default=10, help="Replay FPS")
    parser.add_argument("--reset", action="store_true", help="Reset environment first")
    parser.add_argument("--exec-duration", type=float, default=0.8, help="Trajectory duration")
    
    args = parser.parse_args()

    node = DeltaReplayNode(
        repo_id=args.repo,
        root_dir=args.root,
        episode_idx=args.episode,
        fps=args.fps,
        exec_duration=args.exec_duration
    )
    
    if args.reset:
        node.reset_environment()
    
    # Run replay in separate thread
    thread = threading.Thread(target=node.run_delta_replay, daemon=True)
    thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()