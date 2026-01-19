#!/usr/bin/env python3
"""
Replay Episode Script for ROS/Panda
Loads a LeRobot dataset (local or Hub) and replays the actions on the robot.

IMPORTANT: There are two replay modes:
1. "state" mode (default): Uses the RECORDED STATES from the dataset directly.
   This replays the exact trajectory that was recorded. Best for validation.
   
2. "action" mode: Applies the RECORDED ACTIONS (deltas) to the CURRENT pose.
   Note: The actions in our dataset are "delta to goal" (distance from current
   position to the target), NOT "delta per step". This mode may cause large jumps.

Use "state" mode to verify the recorded data is correct.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import argparse
import time
import threading
import torch
from pathlib import Path

# ROS2 Imports
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import euler_from_quaternion, quaternion_from_euler

# LeRobot Imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class ReplayEpisodeNode(Node):
    def __init__(self, repo_id, root_dir, episode_idx, fps=10, mode="state", 
                 exec_duration=0.8, wait_settle=True):
        super().__init__("replay_episode_node")
        
        self.repo_id = repo_id
        self.root_dir = root_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.control_rate = float(fps)
        self.replay_mode = mode  # "state" or "action"
        self.exec_duration = exec_duration  # How long each trajectory takes
        self.wait_settle = wait_settle  # Whether to wait for arm to settle
        
        # Load Dataset
        self.get_logger().info(f"📂 Loading dataset {repo_id}...")
        self.dataset = LeRobotDataset(repo_id, root=root_dir)
        self.get_logger().info(f"✅ Dataset loaded. Found {self.dataset.num_episodes} episodes.")
        
        if self.episode_idx >= self.dataset.num_episodes:
            self.get_logger().error(f"❌ Episode {self.episode_idx} out of range (max {self.dataset.num_episodes-1})")
            raise ValueError("Episode index out of range")

        # Robot Configuration
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # Data buffers
        self.latest_eef_pose = None
        self.latest_joints = None
        self.data_lock = threading.Lock()
        self.running = False
        
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

    def run_replay(self):
        """Replay the episode loop"""
        self.get_logger().info(f"▶️ Starting Replay of Episode {self.episode_idx} (mode: {self.replay_mode})")
        
        # Get frame indices for this episode
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
        self.get_logger().info(f"⚙️ Exec duration: {self.exec_duration}s, Wait settle: {self.wait_settle}")

        # Wait for initial data
        time.sleep(1.0)
        
        for frame_num, i in enumerate(range(start_idx, end_idx)):
            if not rclpy.ok(): 
                break
            
            cycle_start = time.time()
            
            # Retrieve frame data
            frame = self.dataset[i]
            
            with self.data_lock:
                current_eef = self.latest_eef_pose
                current_joints = self.latest_joints
                
            if current_eef is None or current_joints is None:
                self.get_logger().warn("⚠️ Waiting for robot state...")
                time.sleep(0.1)
                continue

            if self.replay_mode == "state":
                self.replay_from_state(frame, current_joints, frame_num, num_frames)
            else:
                self.replay_from_action(frame, current_eef, current_joints, frame_num, num_frames)
            
            # CRITICAL: Wait for the trajectory to complete before sending the next one
            # This prevents the "wavy" behavior from interrupted trajectories
            if self.wait_settle:
                time.sleep(self.exec_duration + 0.1)  # exec time + small settle buffer
            else:
                # Original behavior: try to maintain FPS (may cause interruptions)
                elapsed = time.time() - cycle_start
                sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
                time.sleep(sleep_time)
            
        self.get_logger().info("🏁 Replay Complete")

    def replay_from_state(self, frame, current_joints, frame_num, total_frames):
        """
        Replay using the RECORDED STATE from the dataset.
        This is the most reliable way to replay - we go directly to the recorded poses.
        """
        # Extract recorded state: [x, y, z, roll, pitch, yaw, grip_l, grip_r]
        state = frame['observation.state']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        x, y, z = state[0], state[1], state[2]
        roll, pitch, yaw = state[3], state[4], state[5]
        gripper_val = state[6]  # -1 = closed, 1 = open
        
        self.get_logger().info(
            f"Frame {frame_num}/{total_frames} | Target XYZ: [{x:.3f}, {y:.3f}, {z:.3f}] | "
            f"Gripper: {'Open' if gripper_val > 0 else 'Closed'}"
        )
        
        # Build target pose from recorded state
        target_quat = quaternion_from_euler(roll, pitch, yaw)
        
        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
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
        else:
            self.get_logger().warn(f"✗ IK failed for frame {frame_num}")

    def replay_from_action(self, frame, current_eef_pose, current_joints, frame_num, total_frames):
        """
        Replay using the RECORDED ACTION (deltas) applied to current pose.
        
        WARNING: Our dataset stores "delta to goal" not "delta per step".
        This means each action is the distance from the current EEF to the target,
        which can be quite large (several cm). This mode may cause large jumps.
        """
        action = frame['action']
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().squeeze()
        
        # Log the action magnitudes
        pos_delta_mag = np.linalg.norm(action[:3])
        self.get_logger().info(
            f"Frame {frame_num}/{total_frames} | Action dXYZ: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}] | "
            f"Magnitude: {pos_delta_mag:.4f}m"
        )
        
        # Check if delta is unreasonably large (e.g., > 5cm in one step at 10Hz)
        if pos_delta_mag > 0.05:
            self.get_logger().warn(
                f"⚠️ Large action delta ({pos_delta_mag:.3f}m)! "
                "This is expected if actions are 'delta-to-goal' not 'delta-per-step'."
            )
        
        # Compute target pose from current + delta
        current_pos = current_eef_pose.position
        current_quat = current_eef_pose.orientation
        current_rpy = euler_from_quaternion([
            current_quat.x, current_quat.y, current_quat.z, current_quat.w
        ])
        
        target_x = float(current_pos.x + action[0])
        target_y = float(current_pos.y + action[1])
        target_z = float(current_pos.z + action[2])
        
        target_rpy = [
            float(current_rpy[0] + action[3]),
            float(current_rpy[1] + action[4]),
            float(current_rpy[2] + action[5])
        ]
        target_quat = quaternion_from_euler(*target_rpy)
        
        # Gripper: -1 = closed, 1 = open, threshold at 0
        gripper_open = bool(action[6] > 0.0)
        self.gripper_pub.publish(Bool(data=gripper_open))
        
        # Build pose
        target_pose = Pose()
        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.position.z = target_z
        target_pose.orientation.x = float(target_quat[0])
        target_pose.orientation.y = float(target_quat[1])
        target_pose.orientation.z = float(target_quat[2])
        target_pose.orientation.w = float(target_quat[3])
        self.target_pub.publish(target_pose)
        
        # Solve IK and execute
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = target_pose
        
        joints = self.get_ik_solution(pose_stamped, current_joints)
        if joints:
            # Use longer duration for large deltas to avoid jerky motion
            duration = max(0.5, pos_delta_mag * 5.0)  # Scale duration with delta size
            self.move_to_joints(joints, duration=duration)
        else:
            self.get_logger().warn("✗ IK failed")
            
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
            self.get_logger().warn("IK timeout")
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
    parser = argparse.ArgumentParser(description="Replay a recorded episode on the robot")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID or local path")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--root", type=str, default=None, help="Root dir for dataset")
    parser.add_argument("--fps", type=int, default=10, help="Replay FPS (only used if --no-wait)")
    parser.add_argument("--reset", action="store_true", help="Reset environment first")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="state", 
        choices=["state", "action"],
        help="Replay mode: 'state' uses recorded poses directly, 'action' applies deltas"
    )
    parser.add_argument(
        "--exec-duration",
        type=float,
        default=0.8,
        help="Duration for each trajectory execution in seconds (default: 0.8)"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for arm to settle between waypoints (causes wavy motion)"
    )
    args = parser.parse_args()

    node = ReplayEpisodeNode(
        repo_id=args.repo,
        root_dir=args.root,
        episode_idx=args.episode,
        fps=args.fps,
        mode=args.mode,
        exec_duration=args.exec_duration,
        wait_settle=not args.no_wait
    )
    
    if args.reset:
        node.reset_environment()
    
    # Run replay in separate thread
    thread = threading.Thread(target=node.run_replay, daemon=True)
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
