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

from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion

# Official LeRobot Imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR

class LeRobotRecorder(Node):
    def __init__(self, repo_id, push_to_hub_flag, gripper_topic, fps, root_dir):
        super().__init__("lerobot_recorder")
        self.bridge = CvBridge()
        self.repo_id = repo_id
        self.push_to_hub_flag = push_to_hub_flag
        self.gripper_topic = gripper_topic
        self.data_lock = threading.Lock()

        # 0. Clean local cache to avoid conflicts
        # Use root_dir if provided, otherwise default to HF cache path
        if root_dir:
            self.root_path = Path(root_dir) / self.repo_id
        else:
            self.root_path = Path.home() / ".cache/huggingface/lerobot" / self.repo_id
            
        if self.root_path.exists():
            shutil.rmtree(self.root_path)

        # 1. Official Dataset Setup
        # We use 'use_videos=True' for better compatibility and smaller file sizes
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=fps,
            root=self.root_path,
            features={
                "observation.images.agentview_image": {
                    "dtype": "video", 
                    "shape": (3, 224, 224), 
                    "names": ["channels", "height", "width"]
                },
                "observation.images.eye_in_hand_image": {
                    "dtype": "video", 
                    "shape": (3, 224, 224), 
                    "names": ["channels", "height", "width"]
                },
                # Changed "vector" to "float32"
                "observation.state": {
                    "dtype": "float32", 
                    "shape": (8,), 
                    "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
                }, 
                "observation.state.joint": {
                    "dtype": "float32", 
                    "shape": (7,), 
                    "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]
                },
                "action": {
                    "dtype": "float32", 
                    "shape": (7,), 
                    "names": ["dx", "dy", "dz", "dr", "dp", "dyaw", "gripper"]
                }, 
            },
            use_videos=True,
            batch_encoding_size=1  # <--- FORCE IMMEDIATE PROCESSING
        )

        # Rate limiting
        self.last_frame_time = 0
        self.target_dt = 1.0 / fps 
        self.frame_count = 0

        # Data Buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_joints = None
        self.latest_eef_pose = None
        self.latest_target_pose = None
        self.latest_gripper_cmd = False
        self.new_side_image = False

        # Subscribers
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.master_shutter_callback, 10)
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_camera_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        self.create_subscription(Pose, "/target_pose_ros", self.target_pose_callback, 10)
        self.create_subscription(Bool, self.gripper_topic, self.gripper_cmd_callback, 10)

        self.is_recording = True
        self.get_logger().info(f"🔴 RECORDING STARTED: {self.repo_id} at 10Hz")

    def master_shutter_callback(self, msg):
        now = time.time()
        if (now - self.last_frame_time) < self.target_dt:
            return
        self.last_frame_time = now
        self.latest_image_top = msg
        self.record_step()

    def side_camera_callback(self, msg):
        with self.data_lock:
            self.latest_image_side = msg
            self.new_side_image = True

    def joint_callback(self, msg): 
        if len(msg.position) >= 7:
            self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    def eef_callback(self, msg): self.latest_eef_pose = msg
    def target_pose_callback(self, msg): self.latest_target_pose = msg
    def gripper_cmd_callback(self, msg): self.latest_gripper_cmd = msg.data

    def record_step(self):
        if not self.is_recording: return

        with self.data_lock:
            # Sync Check: Ensure we have all robot data and a fresh side image
            if not self.new_side_image or any(v is None for v in [self.latest_eef_pose, self.latest_target_pose, self.latest_joints]):
                print("skipping frame")
                return
            
            snap_top = self.latest_image_top
            snap_side = self.latest_image_side
            snap_eef = self.latest_eef_pose
            snap_target = self.latest_target_pose
            snap_joints = self.latest_joints
            snap_gripper = self.latest_gripper_cmd
            self.new_side_image = False

        try:
            # 1. Process Images: ROS (H, W, C) -> LeRobot (C, H, W)
            img_top = self.bridge.compressed_imgmsg_to_cv2(snap_top, "rgb8")
            img_top = cv2.resize(img_top, (224, 224)).transpose(2, 0, 1)
            
            img_side = self.bridge.compressed_imgmsg_to_cv2(snap_side, "rgb8")
            img_side = cv2.resize(img_side, (224, 224)).transpose(2, 0, 1)

            # 2. Calculate State & Action
            gripper_val = 1.0 if snap_gripper else -1.0
            
            state_vec = np.array([
                snap_eef.position.x, snap_eef.position.y, snap_eef.position.z,
                snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w, 
                gripper_val
            ], dtype=np.float32)

            c_rpy = euler_from_quaternion([snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w])
            t_rpy = euler_from_quaternion([snap_target.orientation.x, snap_target.orientation.y, snap_target.orientation.z, snap_target.orientation.w])
            d_rpy = [( (t - c) + np.pi) % (2 * np.pi) - np.pi for t, c in zip(t_rpy, c_rpy)]
            
            action_vec = np.array([
                snap_target.position.x - snap_eef.position.x,
                snap_target.position.y - snap_eef.position.y,
                snap_target.position.z - snap_eef.position.z,
                d_rpy[0], d_rpy[1], d_rpy[2], gripper_val
            ], dtype=np.float32)

            # 3. DIRECT MAPPING (Matches self.dataset.features 1:1)
            # This bypasses the KeyError by using the exact names defined in __init__
            full_frame = {
                "observation.images.agentview_image": img_top.astype(np.uint8),
                "observation.images.eye_in_hand_image": img_side.astype(np.uint8),
                "observation.state": state_vec,
                "observation.state.joint": snap_joints.astype(np.float32),
                "action": action_vec,
                "task": "Pick up the block",
            }
            
            self.dataset.add_frame(full_frame)
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                print(f"Captured {self.frame_count} frames...", end="\r")

        except Exception as e:
            # This will now catch and print the specific error if mapping fails
            self.get_logger().error(f"Recording Error: {e}")

    def stop_episode(self):
        if not self.is_recording: return
        self.is_recording = False
        
        if self.frame_count > 0:
            print(f"\nFinalizing episode with {self.frame_count} frames...")
            # 1. Save the current episode buffer to disk
            self.dataset.save_episode()
            
            # Note: Consolidate is not needed/does not exist in this version of the lib.
            # Stats are computed during save_episode.
            # Pushing to hub is now handled in main() after VideoEncodingManager finalizes.
        else:
            print("\nNo frames were recorded.")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--gripper-topic", type=str, default="/gripper_cmd_aut")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--root", type=str, default=None)
    parsed_args = parser.parse_args([arg for arg in sys.argv[1:] if not arg.startswith('__')])
    
    node = LeRobotRecorder(
        repo_id=parsed_args.repo, 
        push_to_hub_flag=parsed_args.push, 
        gripper_topic=parsed_args.gripper_topic,
        fps=parsed_args.fps,
        root_dir=parsed_args.root
    )
    
    # CRITICAL: Use the VideoEncodingManager context
    # This ensures FFmpeg processes are closed correctly on exit
    with VideoEncodingManager(node.dataset):
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.stop_episode()

    # Push to hub AFTER VideoEncodingManager exits (ensuring finalize() is called)
    if node.push_to_hub_flag and node.frame_count > 0:
        print("Pushing to Hugging Face Hub...")
        node.dataset.push_to_hub()

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()