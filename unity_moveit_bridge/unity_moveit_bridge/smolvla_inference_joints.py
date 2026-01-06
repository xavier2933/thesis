#!/usr/bin/env python3
"""
SmolVLA Joint Inference Script for Panda Arm
Identical to smolvla_inference.py but treats model output as JOINT DELTAS.
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import argparse
import threading
import time
import torch
from pathlib import Path

# ROS2 Imports
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion

# LeRobot Imports (same as inference.py)
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from huggingface_hub import snapshot_download, hf_hub_download


class SmolVLAJointInference(Node):
    def __init__(self, checkpoint_path, task_instruction, device="cuda", stats_path=None, subfolder=None):
        super().__init__("smolvla_joint_inference")
        self.bridge = CvBridge()
        self.device = device
        self.task_instruction = task_instruction
        
        # Load Policy using copy-pasted logic from smolvla_inference.py
        self.get_logger().info(f"🤖 Loading SmolVLA from {checkpoint_path}")
        
        # Handle subfolder for checkpoint weights
        if subfolder and "/" in checkpoint_path:
            self.get_logger().info(f"📂 Downloading subfolder '{subfolder}' from {checkpoint_path}...")
            from pathlib import Path
            repo_path = snapshot_download(repo_id=checkpoint_path)
            config_path = repo_path
            weights_path = str(Path(repo_path) / subfolder / "pretrained_model")
            self.get_logger().info(f"✅ Config from: {config_path}")
            self.get_logger().info(f"✅ Weights from: {weights_path}")
        else:
            config_path = checkpoint_path
            weights_path = checkpoint_path
        
        # Load config
        policy_cfg = PreTrainedConfig.from_pretrained(config_path)
        policy_cfg.device = device
        policy_cfg.pretrained_path = weights_path
        
        # Load metadata
        self.get_logger().info("Loading LIBERO dataset metadata...")
        ds_meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero")
        
        # Load custom stats if provided
        self.action_stats = None
        if stats_path:
            self.get_logger().info(f"Loading action statistics from {stats_path}")
            import json
            if stats_path.startswith("datasets/") or ("/" in stats_path and not Path(stats_path).exists()):
                stats_file = hf_hub_download(repo_id=stats_path, filename="meta/stats.json", repo_type="dataset")
            else:
                stats_file = stats_path
            
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.action_stats = {
                    'mean': np.array(stats['action']['mean'], dtype=np.float32),
                    'std': np.array(stats['action']['std'], dtype=np.float32),
                }
        
        # Create policy
        self.policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
        self.policy.eval()
        
        # Tokenizer fallback
        try:
             # Try processor first (typical for VLMs)
            if hasattr(self.policy.model.vlm_with_expert, 'processor'):
                self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
            elif hasattr(self.policy.model.vlm_with_expert, 'language_model'):
                self.tokenizer = self.policy.model.vlm_with_expert.language_model.tokenizer
            elif hasattr(self.policy.model.vlm_with_expert, 'text_model'):
                self.tokenizer = self.policy.model.vlm_with_expert.text_model.tokenizer
            else:
                from transformers import AutoTokenizer
                vlm_name = policy_cfg.vlm_model_name
                self.tokenizer = AutoTokenizer.from_pretrained(vlm_name)
        except Exception:
            pass # VLM loading handles it usually

        self.get_logger().info("✅ Policy loaded successfully")
        
        # Robot Config
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # Data Buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_eef_pose = None
        self.latest_joints = None
        self.new_data = False
        self.data_lock = threading.Lock()
        
        # Publishers
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.gripper_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        
        # Subscribers
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.top_camera_callback, 10)
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_camera_callback, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        
        self.control_rate = 10.0

    # --- CALLBACKS ---
    def top_camera_callback(self, msg):
        with self.data_lock:
            self.latest_image_top = msg
            self.new_data = True
            
    def side_camera_callback(self, msg):
        with self.data_lock: self.latest_image_side = msg
            
    def eef_callback(self, msg):
        with self.data_lock: self.latest_eef_pose = msg
    
    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            with self.data_lock:
                self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    # --- INFERENCE LOOP ---
    def run_inference_loop(self):
        last_cycle_start = time.time()
        
        while rclpy.ok():
            now = time.time()
            cycle_dt = now - last_cycle_start
            last_cycle_start = now
            
            # Adaptive duration
            exec_duration = max(0.5, cycle_dt * 1.2)
            
            with self.data_lock:
                if not self.new_data or any(v is None for v in [self.latest_image_top, self.latest_image_side, self.latest_eef_pose, self.latest_joints]):
                    time.sleep(0.01)
                    continue
                snap_top, snap_side = self.latest_image_top, self.latest_image_side
                snap_eef, snap_joints = self.latest_eef_pose, self.latest_joints
                self.new_data = False
            
            try:
                # 1. Observation
                observation = self.prepare_observation(snap_top, snap_side, snap_eef)
                
                # 2. Inference
                with torch.no_grad():
                    action_dict = self.policy.select_action(observation)
                    if isinstance(action_dict, dict):
                        action = action_dict.get("action", action_dict.get("actions"))
                    else:
                        action = action_dict
                
                # 3. Execute as JOINT DELTAS
                self.execute_joint_action(action, snap_joints, duration=exec_duration)
                
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
            
            elapsed = time.time() - now
            sleep_time = max(0.0, (1.0/self.control_rate) - elapsed)
            time.sleep(sleep_time)

    def prepare_observation(self, img_top, img_side, eef_pose):
        # Decode and Resize
        i1 = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(img_top, "rgb8"), (256, 256)).transpose(2,0,1)
        i2 = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(img_side, "rgb8"), (256, 256)).transpose(2,0,1)
        
        # State Vector (8D: Pose + Gripper)
        # Note: We send EEF state, even though output is Joint Deltas.
        pos, quat = eef_pose.position, eef_pose.orientation
        rpy = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        # Placeholder gripper
        state_vec = np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2], 1.0, -1.0], dtype=np.float32)

        # Tokenize the task instruction
        tokens = self.tokenizer(
            self.task_instruction,
            padding="max_length",
            truncation=True,
            max_length=48,  # Default max length for SmolVLA
            return_tensors="pt"
        )
        
        return {
            "observation.images.image": torch.from_numpy(i1).float().div(255.0).unsqueeze(0).to(self.device),
            "observation.images.image2": torch.from_numpy(i2).float().div(255.0).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(state_vec).unsqueeze(0).to(self.device),
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(self.device),
        }

    def execute_joint_action(self, action, current_joints, duration=0.1):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().squeeze()
            
        # Denormalize? 
        # smolvla_inference.py does NOT denormalize explicitly here because LeRobot policy handles it usually.
        # But if you have action_stats loaded, you might need to. 
        # For now, assuming Policy returns unnormalized action.
        
        # Action is [dJ1...dJ7] (and maybe gripper)
        joints_delta = action[:7]
        target_joints = current_joints + joints_delta
        
        # Publish
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target_joints]  # Convert numpy to standard float list
        point.velocities = [0.0] * 7
        point.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        traj.points = [point]
        self.arm_pub.publish(traj)
        
        # Gripper
        if len(action) > 7:
            gripper_open = bool(action[7] > 0.0)
            self.gripper_pub.publish(Bool(data=gripper_open))

def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="Pick up the block")
    parser.add_argument("--stats", type=str, default=None)
    args = parser.parse_args()
    
    node = SmolVLAJointInference(args.repo, args.task, args.device, stats_path=args.stats)
    
    t = threading.Thread(target=node.run_inference_loop, daemon=True)
    t.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
