#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import argparse
import sys
import threading
import time
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil

class LeRobotRecorder(Node):
    def __init__(self, repo_id, push_to_hub_flag, gripper_topic):
        super().__init__("lerobot_recorder")
        self.bridge = CvBridge()
        self.repo_id = repo_id
        self.push_to_hub_flag = push_to_hub_flag
        self.gripper_topic = gripper_topic
        self.data_lock = threading.Lock()

        # 0. Clean local cache
        local_dir = Path.home() / ".cache/huggingface/lerobot" / self.repo_id
        if local_dir.exists():
            shutil.rmtree(local_dir)

        # 1. Dataset Setup - Matches your 'help' output exactly
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=10,
            features={
                "observation.images.agentview_image": {
                    "dtype": "image", 
                    "shape": (3, 224, 224), 
                    "names": ["channels", "height", "width"]
                },
                "observation.images.eye_in_hand_image": {
                    "dtype": "image", 
                    "shape": (3, 224, 224), 
                    "names": ["channels", "height", "width"]
                },
                "observation.state": {"dtype": "float32", "shape": (8,)}, 
                "observation.state.joint": {"dtype": "float32", "shape": (7,)},
                "action": {"dtype": "float32", "shape": (7,)}, 
            },
            use_videos=False  # Explicitly tell it we aren't using video
        )

        # Rate limiting (10Hz)
        self.last_frame_time = 0
        self.target_dt = 1.0 / 10.0 
        self.frame_count = 0

        # Data Buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_joints = None
        self.latest_eef_pose = None
        self.latest_target_pose = None
        self.latest_gripper_cmd = False
        
        # Sync Flag
        self.new_side_image = False

        # Subscribers
        # MASTER: The top camera drives the 10Hz clock
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.master_shutter_callback, 10)
        
        # SLAVES: These just update the buffers
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_camera_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        self.create_subscription(Pose, "/target_pose_ros", self.target_pose_callback, 10)
        self.create_subscription(Bool, self.gripper_topic, self.gripper_cmd_callback, 10)

        self.is_recording = True
        self.get_logger().info(f"🔴 RECORDING TO: {self.repo_id} at 10Hz")

    def master_shutter_callback(self, msg):
        """THROTTLED Heartbeat: Only triggers record_step 10 times per second"""
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
                return
            
            snap_top = self.latest_image_top
            snap_side = self.latest_image_side
            snap_eef = self.latest_eef_pose
            snap_target = self.latest_target_pose
            snap_joints = self.latest_joints
            snap_gripper = self.latest_gripper_cmd
            self.new_side_image = False

        try:
            # Prepare Images
            frame_top = self.bridge.compressed_imgmsg_to_cv2(snap_top, "rgb8")
            frame_side = self.bridge.compressed_imgmsg_to_cv2(snap_side, "rgb8")
            frame_top = cv2.resize(frame_top, (224, 224)).transpose(2, 0, 1)
            frame_side = cv2.resize(frame_side, (224, 224)).transpose(2, 0, 1)

            # Calculate State and Action
            gripper_val = 1.0 if snap_gripper else -1.0
            state_vec = [snap_eef.position.x, snap_eef.position.y, snap_eef.position.z,
                         snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w, 
                         gripper_val]

            dx = snap_target.position.x - snap_eef.position.x
            dy = snap_target.position.y - snap_eef.position.y
            dz = snap_target.position.z - snap_eef.position.z
            
            c_rpy = euler_from_quaternion([snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w])
            t_rpy = euler_from_quaternion([snap_target.orientation.x, snap_target.orientation.y, snap_target.orientation.z, snap_target.orientation.w])
            d_rpy = [( (t - c) + np.pi) % (2 * np.pi) - np.pi for t, c in zip(t_rpy, c_rpy)]
            
            action_vec = [dx, dy, dz, d_rpy[0], d_rpy[1], d_rpy[2], gripper_val]

            # Write data
            self.dataset.add_frame({
                "observation.images.agentview_image": frame_top,
                "observation.images.eye_in_hand_image": frame_side,
                "observation.state": np.array(state_vec, dtype=np.float32),
                "observation.state.joint": snap_joints,
                "action": np.array(action_vec, dtype=np.float32),
                "task": "Pick up the block",
            })
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                print(f"Captured {self.frame_count} frames...", end="\r")

        except Exception as e:
            self.get_logger().error(f"Recording Error: {e}")

    def stop_episode(self):
        if not self.is_recording: return
        self.is_recording = False
        print("\nStopping... Saving episode and finishing video encoding.")
        self.dataset.save_episode()
        # In your version, finalize might take time if chunks are being consolidated
        self.dataset.finalize()
        if self.push_to_hub_flag:
            print("Pushing to Hub...")
            self.dataset.push_to_hub()

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--gripper-topic", type=str, default="/gripper_cmd_aut")
    args = parser.parse_args([arg for arg in sys.argv[1:] if not arg.startswith('__')])
    
    node = LeRobotRecorder(repo_id=args.repo, push_to_hub_flag=args.push, gripper_topic=args.gripper_topic)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_episode()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()