#!/usr/bin/env python3
"""
SmolVLA Fast Inference Script
Actions sent at max(10Hz, inference_speed) - no artificial delays.
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

from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from huggingface_hub import snapshot_download, hf_hub_download


class SmolVLAFastInference(Node):
    def __init__(self, checkpoint_path, task_instruction, device="cuda", stats_path=None, subfolder=None, n_action_steps=50):
        super().__init__("smolvla_fast_inference")
        self.bridge = CvBridge()
        self.device = device
        self.task_instruction = task_instruction
        
        # Load Policy
        self.get_logger().info(f"🤖 Loading SmolVLA from {checkpoint_path}")
        
        # Handle subfolder for checkpoint weights
        if subfolder and "/" in checkpoint_path:
            self.get_logger().info(f"📂 Downloading subfolder '{subfolder}' from {checkpoint_path}...")
            repo_path = snapshot_download(repo_id=checkpoint_path)
            config_path = repo_path
            weights_path = str(Path(repo_path) / subfolder / "pretrained_model")
            self.get_logger().info(f"✅ Config from: {config_path}")
            self.get_logger().info(f"✅ Weights from: {weights_path}")
        else:
            config_path = checkpoint_path
            weights_path = checkpoint_path
        
        policy_cfg = PreTrainedConfig.from_pretrained(config_path)
        policy_cfg.device = device
        policy_cfg.pretrained_path = weights_path
        policy_cfg.n_action_steps = n_action_steps  # Override action steps
        
        self.get_logger().info(f"📏 Using n_action_steps={n_action_steps} (model re-queries every {n_action_steps} steps)")
        
        ds_meta = LeRobotDatasetMetadata("Xavier033/new_libero")
        
        # Load stats for denormalization
        self.action_stats = None
        if stats_path:
            self.get_logger().info(f"Loading action stats from {stats_path}")
            import json
            if "/" in stats_path and not Path(stats_path).exists():
                stats_file = hf_hub_download(repo_id=stats_path, filename="meta/stats.json", repo_type="dataset")
            else:
                stats_file = stats_path
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.action_stats = {
                    'mean': np.array(stats['action']['mean'], dtype=np.float32),
                    'std': np.array(stats['action']['std'], dtype=np.float32),
                }
        
        self.policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
        self.policy.eval()
        
        # Get tokenizer
        if hasattr(self.policy.model.vlm_with_expert, 'processor'):
            self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(policy_cfg.vlm_model_name)
        
        self.get_logger().info("✅ Policy loaded")
        
        # Robot config
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # Data buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_eef_pose = None
        self.latest_joints = None
        self.data_lock = threading.Lock()
        self.new_data = False
        
        # Publishers
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        self.gripper_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        
        # Subscribers
        self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.top_cb, 10)
        self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_cb, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
        
        # IK Service
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        while not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for IK...")

    def top_cb(self, msg):
        with self.data_lock:
            self.latest_image_top = msg
            self.new_data = True
            
    def side_cb(self, msg):
        with self.data_lock:
            self.latest_image_side = msg
            
    def eef_cb(self, msg):
        with self.data_lock:
            self.latest_eef_pose = msg
    
    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            with self.data_lock:
                self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    def run_inference_loop(self):
        """Run as fast as possible, capped at 10Hz"""
        min_dt = 0.1  # 10Hz max
        
        while rclpy.ok():
            loop_start = time.time()
            
            with self.data_lock:
                if not self.new_data or any(v is None for v in [
                    self.latest_image_top, self.latest_image_side, 
                    self.latest_eef_pose, self.latest_joints
                ]):
                    time.sleep(0.01)
                    continue
                snap_top = self.latest_image_top
                snap_side = self.latest_image_side
                snap_eef = self.latest_eef_pose
                snap_joints = self.latest_joints
                self.new_data = False
            
            try:
                # Inference
                t0 = time.time()
                observation = self.prepare_observation(snap_top, snap_side, snap_eef)
                with torch.no_grad():
                    action_dict = self.policy.select_action(observation)
                    action = action_dict.get("action", action_dict) if isinstance(action_dict, dict) else action_dict
                inference_time = time.time() - t0
                
                # Execute immediately - trajectory duration = time until next action
                # Use min_dt (0.1s) as the trajectory duration
                self.execute_action(action, snap_eef, snap_joints, duration=min_dt)
                
                self.get_logger().info(f"Inference: {inference_time:.3f}s")
                
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
            
            # Sleep only if faster than 10Hz
            elapsed = time.time() - loop_start
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)

    def prepare_observation(self, img_top_msg, img_side_msg, eef_pose):
        img_top = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(img_top_msg, "rgb8"), (256, 256)).transpose(2, 0, 1)
        img_side = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(img_side_msg, "rgb8"), (256, 256)).transpose(2, 0, 1)
        
        pos, quat = eef_pose.position, eef_pose.orientation
        rpy = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        state = np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2], 1.0, -1.0], dtype=np.float32)
        
        tokens = self.tokenizer(self.task_instruction, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
        
        return {
            "observation.images.image": torch.from_numpy(img_top).float().unsqueeze(0).to(self.device) / 255.0,
            "observation.images.image2": torch.from_numpy(img_side).float().unsqueeze(0).to(self.device) / 255.0,
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(self.device),
        }

    def execute_action(self, action, current_eef_pose, current_joints, duration=0.1):
        action = action.cpu().numpy().squeeze()
        
        # Denormalize if stats provided
        if self.action_stats is not None:
            action = action * self.action_stats['std'] + self.action_stats['mean']
        
        # Compute target pose
        current_pos = current_eef_pose.position
        current_quat = current_eef_pose.orientation
        current_rpy = euler_from_quaternion([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        
        target_x = float(current_pos.x + action[0])
        target_y = float(current_pos.y + action[1])
        target_z = float(current_pos.z + action[2])
        target_rpy = [float(current_rpy[i] + action[3+i]) for i in range(3)]
        target_quat = quaternion_from_euler(*target_rpy)
        
        # Gripper
        self.gripper_pub.publish(Bool(data=bool(action[6] > 0.0)))
        
        # Build pose
        target_pose = Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = target_x, target_y, target_z
        target_pose.orientation.x, target_pose.orientation.y = float(target_quat[0]), float(target_quat[1])
        target_pose.orientation.z, target_pose.orientation.w = float(target_quat[2]), float(target_quat[3])
        self.target_pub.publish(target_pose)
        
        # IK and execute
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = target_pose
        
        joints = self.get_ik_solution(pose_stamped)
        if joints:
            self.move_to_joints(joints, duration=duration)

    def get_ik_solution(self, pose_stamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        event = threading.Event()
        result = {"res": None}
        def cb(f): result["res"] = f.result(); event.set()
        self.ik_client.call_async(req).add_done_callback(cb)
        
        if not event.wait(timeout=1.0):
            return None
        if result["res"] and result["res"].error_code.val == MoveItErrorCodes.SUCCESS:
            return list(result["res"].solution.joint_state.position[:7])
        return None

    def move_to_joints(self, joints, duration=0.1):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = list(joints)
        pt.velocities = [0.0] * 7
        pt.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        traj.points = [pt]
        self.arm_pub.publish(traj)

    def reset_environment(self):
        self.pub_aut.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_reset.publish(Bool(data=True))
        time.sleep(4.0)
        self.pub_aut.publish(Bool(data=True))
        time.sleep(0.5)


def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--task", type=str, default="Pick up the blue block")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder in HF repo (e.g., 'checkpoints/003000')")
    parser.add_argument("--n_action_steps", type=int, default=50, help="Actions to execute before re-querying model (lower = more reactive, e.g., 10)")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    
    node = SmolVLAFastInference(args.checkpoint, args.task, args.device, args.stats, args.subfolder, args.n_action_steps)
    
    if args.reset:
        node.reset_environment()
    
    node.get_logger().info("🚀 Running at max 10Hz (or inference speed)")
    
    thread = threading.Thread(target=node.run_inference_loop, daemon=True)
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
