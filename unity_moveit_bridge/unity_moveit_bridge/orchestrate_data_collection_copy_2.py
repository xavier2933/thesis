#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import argparse
import sys
import threading
import time
import shutil
from pathlib import Path

# ROS2/Math Imports
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_multiply, quaternion_from_euler
import tf2_ros

# LeRobot Imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
import random

class SmolVLAOrchestrator(Node):
    def __init__(self, repo_id, fps, root_dir, total_episodes):
        super().__init__("smol_vla_orchestrator")
        self.bridge = CvBridge()
        self.repo_id = repo_id
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.data_lock = threading.Lock()
        self.is_recording = False

        # --- 1. Dataset Setup ---
        if root_dir:
            self.root_path = Path(root_dir) / self.repo_id
        else:
            self.root_path = Path.home() / ".cache/huggingface/lerobot" / self.repo_id
        
        # Check if dataset already exists - if so, load it to add more episodes
        if self.root_path.exists():
            self.get_logger().info(f"📂 Loading existing dataset from {self.root_path}")
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root_path)
            self.get_logger().info(f"✅ Loaded {self.dataset.num_episodes} existing episodes")
        else:
            self.get_logger().info(f"📂 Creating new dataset at {self.root_path}")
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=fps,
                root=self.root_path,
                features = {
                    "observation.images.image": {
                        "dtype": "image",
                        "shape": [3, 256, 256],
                        "names": ["channels", "height", "width"],
                    },
                    "observation.images.image2": {
                        "dtype": "image",
                        "shape": [3, 256, 256],
                        "names": ["channels", "height", "width"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["x", "y", "z", "roll", "pitch", "yaw", "grip_l", "grip_r"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"],
                    },
                },
                use_videos=True
            )

        self.instructions = [
            "Pick up the blue block",
            "Grasp the blue object and lift it",
            "Retrieve the blue object",
            "Pick up the blue item"
        ]
        self.current_task = "Pick up the blue block"

        # --- 2. Motion/Control Config ---
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.target_frame = "block"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        # --- 3. Publishers & Subscribers ---
        # Publishers
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        self.gripper_aut_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)

        # Buffers for recording
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_joints = None
        self.latest_eef_pose = None
        self.latest_target_pose = None
        self.latest_gripper_cmd = False
        self.new_side_image = False
        self.last_frame_time = 0
        self.target_dt = 1.0 / fps

        # Recording Subs
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.master_shutter_callback, 10)
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_camera_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        self.create_subscription(Pose, "/target_pose_ros", self.target_pose_callback, 10)
        self.create_subscription(Bool, "/gripper_cmd_aut", self.gripper_cmd_callback, 10)

        self.create_subscription(Bool, "/gripper_cmd_aut", self.gripper_cmd_callback, 10)

    def refresh_tf(self):
        """Re-initializes TF buffer to handle simulation time resets."""
        self.get_logger().info("🔄 Refreshing TF Buffer...")
        # Manually unregister old listener if possible (Python API is limited here, relying on GC)
        if hasattr(self, 'tf_listener'):
            del self.tf_listener
        if hasattr(self, 'tf_buffer'):
            self.tf_buffer.clear()
            del self.tf_buffer
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Wait a moment for buffer to fill
        time.sleep(1.0)

    # --- RECORDING CALLBACKS ---
    def master_shutter_callback(self, msg):
        now = time.time()
        if not self.is_recording or (now - self.last_frame_time) < self.target_dt:
            return
        self.last_frame_time = now
        self.latest_image_top = msg
        self.record_step()

    def side_camera_callback(self, msg):
        with self.data_lock:
            self.latest_image_side = msg
            self.new_side_image = True

    def joint_callback(self, msg): 
        if len(msg.position) >= 7: self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    def eef_callback(self, msg): self.latest_eef_pose = msg
    def target_pose_callback(self, msg): self.latest_target_pose = msg
    def gripper_cmd_callback(self, msg): self.latest_gripper_cmd = msg.data

    def record_step(self):
        with self.data_lock:
            if not self.is_recording:
                return
            # Ensure we have all necessary data before proceeding
            if not self.new_side_image or any(v is None for v in [self.latest_eef_pose, self.latest_target_pose, self.latest_joints]):
                return
            
            # Snap data
            snap_top, snap_side = self.latest_image_top, self.latest_image_side
            snap_eef, snap_target = self.latest_eef_pose, self.latest_target_pose
            snap_joints, snap_gripper = self.latest_joints, self.latest_gripper_cmd
            self.new_side_image = False

        try:
            # 1. Process Images: Resize to 256x256 as per config.json
            img_top = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(snap_top, "rgb8"), (256, 256)).transpose(2, 0, 1)
            img_side = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(snap_side, "rgb8"), (256, 256)).transpose(2, 0, 1)

            # 2. Logic for Rotation and Gripper
            gripper_val = 1.0 if snap_gripper else -1.0  # -1 = Closed, 1 = Open
            
            # Current End-Effector RPY
            c_rpy = euler_from_quaternion([snap_eef.orientation.x, snap_eef.orientation.y, 
                                        snap_eef.orientation.z, snap_eef.orientation.w])
            
            # 3. State Vector: 8-DOF (XYZ, RPY, Finger1, Finger2)
            state_vec = np.array([
                snap_eef.position.x, snap_eef.position.y, snap_eef.position.z,
                c_rpy[0], c_rpy[1], c_rpy[2], 
                gripper_val, -gripper_val
            ], dtype=np.float32)

            # 🌟 4. Compute Action as DELTA BETWEEN CONSECUTIVE STATES
            if not hasattr(self, 'prev_eef_pose') or self.prev_eef_pose is None:
                # First frame: zero action (no previous state)
                action_vec = np.zeros(7, dtype=np.float32)
                action_vec[6] = gripper_val  # Include gripper state
                self.get_logger().info("First frame: using zero action")
            else:
                # Compute delta from previous to current state
                prev_rpy = euler_from_quaternion([
                    self.prev_eef_pose.orientation.x,
                    self.prev_eef_pose.orientation.y,
                    self.prev_eef_pose.orientation.z,
                    self.prev_eef_pose.orientation.w
                ])
                
                # Position delta
                dx = snap_eef.position.x - self.prev_eef_pose.position.x
                dy = snap_eef.position.y - self.prev_eef_pose.position.y
                dz = snap_eef.position.z - self.prev_eef_pose.position.z
                
                # Rotation delta (shortest path)
                d_rpy = [
                    ((c_rpy[i] - prev_rpy[i]) + np.pi) % (2 * np.pi) - np.pi
                    for i in range(3)
                ]
                
                action_vec = np.array([
                    dx, dy, dz,
                    d_rpy[0], d_rpy[1], d_rpy[2],
                    gripper_val
                ], dtype=np.float32)
                
                # Log action magnitude for debugging
                pos_mag = np.linalg.norm([dx, dy, dz])
                if pos_mag > 0.05:  # More than 5cm in one step at 10Hz
                    self.get_logger().warn(f"Large action delta: {pos_mag:.4f}m - possible issue!")

            # Store current pose for next iteration
            import copy
            self.prev_eef_pose = copy.deepcopy(snap_eef)

            # 5. Save to LeRobot Dataset
            self.dataset.add_frame({
                "observation.images.image": img_top.astype(np.uint8),
                "observation.images.image2": img_side.astype(np.uint8),
                "observation.state": state_vec,
                "action": action_vec,
                "task": self.current_task,
            })

        except Exception as e:
            self.get_logger().error(f"Record Error: {e}")
            import traceback
            traceback.print_exc()

    # --- ROBOT CONTROL LOGIC ---
    def reset_environment(self):
        self.get_logger().info(f"🔄 Resetting Environment (Episode {self.current_episode + 1}/{self.total_episodes})")
        self.pub_aut.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_reset.publish(Bool(data=True))
        time.sleep(4.0)
        self.pub_aut.publish(Bool(data=True))
        time.sleep(0.5)

    def control_gripper(self, open_gripper: bool):
        msg = Bool(data=open_gripper)
        self.gripper_aut_pub.publish(msg)
        time.sleep(1.0)

    def get_ik_solution(self, pose: PoseStamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
    
        # 🌟 CRITICAL: Seed with current joints to prevent elbow flipping
        # if self.latest_joints is not None:
        #     req.ik_request.robot_state.joint_state.name = self.joint_names
        #     req.ik_request.robot_state.joint_state.position = self.latest_joints.tolist()
        
        # Create an event to block this thread without spinning ROS
        event = threading.Event()
        result_wrapper = {"res": None}

        def done_callback(future):
            result_wrapper["res"] = future.result()
            event.set()

        future = self.ik_client.call_async(req)
        future.add_done_callback(done_callback)
        
        if not event.wait(timeout=2.0):
            self.get_logger().error("IK service timed out")
            return None
            
        res = result_wrapper["res"]
        if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
            return res.solution.joint_state.position[:7]
        
        self.get_logger().warn(f"IK failed with code: {res.error_code.val if res else 'No Response'}")
        return None

    def move_to_pose(self, x, y, z, q, duration):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        
        self.target_pub.publish(pose.pose)
        joints = self.get_ik_solution(pose)
        if joints:
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            pt = JointTrajectoryPoint(positions=joints, time_from_start=rclpy.duration.Duration(seconds=duration).to_msg())
            traj.points = [pt]
            self.arm_pub.publish(traj)
            time.sleep(duration + 0.5)
        else:
            self.get_logger().error("IK Failed")

    def run_collection(self):
        with VideoEncodingManager(self.dataset):
            i = 0
            while i < self.total_episodes:
                self.get_logger().info(f"🤖 Starting Episode {i+1}/{self.total_episodes}")
                self.get_logger().info(f"📋 Task: {self.current_task}")
                
                self.current_episode = i
                self.reset_environment()
                self.prev_eef_pose = None

                # Start Recording
                with self.data_lock:
                    self.is_recording = True
                
                self.get_logger().info(f"🔴 Recording...")

                motion_success = True
                try:
                    # --- Your Execution Sequence ---
                    self.refresh_tf() # 🌟 Clear TF buffer to accept timestamps from T=0 (post-reset)
                    self.control_gripper(True)

                    # Simple lookup - the buffer is fresh, so whatever we find is new
                    tf = self.tf_buffer.lookup_transform(self.base_frame, self.target_frame, rclpy.time.Time())
                    q_block = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                    q_block = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                    q_correction = quaternion_from_euler(-1.57079632679, 0.0, 0.0)
                    q_fixed = quaternion_multiply(q_block, q_correction)
                    
                    tx, ty, tz = tf.transform.translation.x + 0.04, tf.transform.translation.y - 0.012, tf.transform.translation.z
                    self.move_to_pose(tx, ty, tz + 0.13, q_fixed, 3.0) 
                    self.move_to_pose(tx, ty, tz + 0.07, q_fixed, 3.0) 
                    self.control_gripper(False) 
                    self.move_to_pose(tx, ty, tz + 0.15, q_fixed, 2.0) 
                except Exception as e:
                    self.get_logger().error(f"❌ Motion error: {e}")
                    motion_success = False

                # Stop Recording
                with self.data_lock:
                    self.is_recording = False

                # --- QUALITY CONTROL PROMPT ---
                if motion_success:
                    print(f"\n--- Episode {i+1} Finished ---")
                    print(f"Frames recorded: 69")
                    user_input = input("Save this episode? [y]es / [n]o (discard): ").lower().strip()
                    
                    if user_input == 'y':
                        with self.data_lock:
                            self.dataset.save_episode()
                        self.get_logger().info(f"✅ Episode {i+1} saved to disk.")
                        i += 1 # Only increment successful count if saved
                    else:
                        # Clear the internal buffer without saving to disk
                        self.dataset.clear_episode_buffer()
                        self.get_logger().warn(f"🗑️ Episode {i+1} discarded by user.")
                else:
                    self.dataset.clear_episode_buffer()
                    self.get_logger().error(f"⚠️ Episode failed or empty. Discarded automatically.")
                
                print("-" * 30)

        self.get_logger().info("🎉 Collection Complete!")
        if self.repo_id: 
            print("Pushing to Hugging Face Hub...")
            self.dataset.push_to_hub()

def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    # Xavier033/pick_place_LIBERO
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    node = SmolVLAOrchestrator(repo_id=args.repo, fps=10, root_dir=None, total_episodes=args.episodes)
    
    # Run loop in a separate thread so ROS callbacks stay alive
    thread = threading.Thread(target=node.run_collection, daemon=True)
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