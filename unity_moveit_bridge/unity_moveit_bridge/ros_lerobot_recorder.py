#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import argparse
import sys
from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from tf_transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class LeRobotRecorder(Node):
    def __init__(self, repo_id, push_to_hub_flag):
        super().__init__("lerobot_recorder")
        self.bridge = CvBridge()
        self.repo_id = repo_id
        self.push_to_hub_flag = push_to_hub_flag
        
        # 1. Setup LeRobot Dataset
        # Note: We define the features we want to record. 
        # Actions are 7-dim: Delta Position (3) + Delta Rotation (3, Euler/Axis-Angle) + Gripper (1)
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=10,
            features={
                "observation.images.camera_top": {"dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
                "observation.images.camera_side": {"dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
                "observation.state": {"dtype": "float32", "shape": (8,)}, # EEF Pose(7: x,y,z,qx,qy,qz,qw) + Gripper(1)
                "observation.state.joint": {"dtype": "float32", "shape": (7,)},
                "action": {"dtype": "float32", "shape": (7,)}, # Delta: dx, dy, dz, droll, dpitch, dyaw, gripper
            }
        )

        # 2. State Buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_joints = None
        self.latest_eef_pose = None # Pose object
        
        # Inputs from Unity/Teleop
        self.latest_target_pose = None # Pose object
        self.latest_wrist_angle = 0.0
        self.latest_gripper_cmd = False # False=Closed, True=Opened (unity logic)

        # 3. Subscribers
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.image_top_callback, 10)
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.image_side_callback, 10)
        
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        
        # Subscribe to standard ROS-frame target pose (from test scripts or planners)
        self.create_subscription(Pose, "/target_pose_ros", self.target_pose_callback, 10)
        
        # For gripper, we can listen to the boolean command or just a float state if provided
        self.create_subscription(Bool, "/gripper_command", self.gripper_cmd_callback, 10)

        # 4. Timer for Recording Loop (10Hz)
        self.create_timer(0.1, self.record_step)
        
        self.is_recording = False
        self.get_logger().info(f"RECORDER READY. Repository: {self.repo_id}")
        self.get_logger().info("Use keyboard interrupt (Ctrl+C) to stop and save.")
        
        self.start_episode()

    # --- Callbacks ---
    def image_top_callback(self, msg): self.latest_image_top = msg
    def image_side_callback(self, msg): self.latest_image_side = msg
    def joint_callback(self, msg): 
        if len(msg.position) >= 7:
            self.latest_joints = np.array(msg.position[:7], dtype=np.float32)
            
    def eef_callback(self, msg): self.latest_eef_pose = msg
    
    def target_pose_callback(self, msg): self.latest_target_pose = msg
    def gripper_cmd_callback(self, msg): self.latest_gripper_cmd = msg.data

    # --- Recording Logic ---
    def start_episode(self):
        self.is_recording = True
        self.get_logger().info("Recording started...")

    def record_step(self):
        if not self.is_recording:
            return
            
        if (self.latest_image_top is None or 
            self.latest_image_side is None or 
            self.latest_eef_pose is None or 
            self.latest_target_pose is None or 
            self.latest_joints is None):
            return

        # 1. Process Images
        try:
            frame_top = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_top, "rgb8")
            frame_side = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_side, "rgb8")
            frame_top = cv2.resize(frame_top, (224, 224)).transpose(2, 0, 1)
            frame_side = cv2.resize(frame_side, (224, 224)).transpose(2, 0, 1)
        except Exception as e:
            self.get_logger().warn(f"Image processing failed: {e}")
            return

        # 2. Process State
        gripper_state = 1.0 if self.latest_gripper_cmd else 0.0
        eef = self.latest_eef_pose
        
        state_vec = [
            eef.position.x, eef.position.y, eef.position.z,
            eef.orientation.x, eef.orientation.y, eef.orientation.z, eef.orientation.w,
            gripper_state
        ]

        # 3. Calculate Action (Target Delta)
        # Using direct ROS frame subtraction
        target = self.latest_target_pose
        
        dx = target.position.x - eef.position.x
        dy = target.position.y - eef.position.y
        dz = target.position.z - eef.position.z
        
        # Orientation Delta (Euler RPY)
        current_q = [eef.orientation.x, eef.orientation.y, eef.orientation.z, eef.orientation.w]
        target_q = [target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w]
        
        current_rpy = euler_from_quaternion(current_q)
        target_rpy = euler_from_quaternion(target_q)
        
        d_roll = target_rpy[0] - current_rpy[0]
        d_pitch = target_rpy[1] - current_rpy[1]
        d_yaw = target_rpy[2] - current_rpy[2]
        
        # Normalize angles
        d_roll = (d_roll + np.pi) % (2 * np.pi) - np.pi
        d_pitch = (d_pitch + np.pi) % (2 * np.pi) - np.pi
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
        
        action_vec = [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_state]

        # 4. Add to Dataset
        self.dataset.add_frame({
            "observation.images.camera_top": frame_top,
            "observation.images.camera_side": frame_side,
            "observation.state": np.array(state_vec, dtype=np.float32),
            "observation.state.joint": self.latest_joints,
            "action": np.array(action_vec, dtype=np.float32),
        })

    def stop_episode(self):
        self.is_recording = False
        self.get_logger().info("Saving episode...")
        self.dataset.save_episode()
        self.get_logger().info("Episode saved.")
        
        if self.push_to_hub_flag:
            self.get_logger().info(f"Pushing to Hub: {self.repo_id}...")
            try:
                self.dataset.push_to_hub() 
                self.get_logger().info("✅ Push successful!")
            except Exception as e:
                self.get_logger().error(f"❌ Push to Hub failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="xavie/green_arm_block_pick", help="HuggingFace Repo ID")
    parser.add_argument("--push", action="store_true", help="Push to Hub after recording")
    
    # Filter out ROS args
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('__')]
    args = parser.parse_args(filtered_args)
    
    node = LeRobotRecorder(repo_id=args.repo, push_to_hub_flag=args.push)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_episode()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()