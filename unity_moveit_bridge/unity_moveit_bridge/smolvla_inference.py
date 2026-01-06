#!/usr/bin/env python3
"""
SmolVLA Inference Script for Panda Arm
Runs the LIBERO-trained model checkpoint for comparison/evaluation
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
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros

# LeRobot Imports (new API)
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from huggingface_hub import snapshot_download, hf_hub_download


class SmolVLAInference(Node):
    def __init__(self, checkpoint_path, task_instruction, device="cuda", stats_path=None, subfolder=None):
        super().__init__("smolvla_inference")
        self.bridge = CvBridge()
        self.device = device
        self.task_instruction = task_instruction
        
        # Load Policy using new API
        self.get_logger().info(f"🤖 Loading SmolVLA from {checkpoint_path}")
        
        # Handle subfolder for checkpoint weights
        original_checkpoint = checkpoint_path
        if subfolder and "/" in checkpoint_path:
            self.get_logger().info(f"📂 Downloading subfolder '{subfolder}' from {checkpoint_path}...")
            from pathlib import Path
            
            # Download both root (for config) and checkpoint subfolder (for weights)
            repo_path = snapshot_download(repo_id=checkpoint_path)
            
            # Config is at root
            config_path = repo_path
            # Weights are in subfolder/pretrained_model/
            weights_path = str(Path(repo_path) / subfolder / "pretrained_model")
            
            self.get_logger().info(f"✅ Config from: {config_path}")
            self.get_logger().info(f"✅ Weights from: {weights_path}")
        else:
            # No subfolder - everything at root
            config_path = checkpoint_path
            weights_path = checkpoint_path
        
        # Load config from root
        policy_cfg = PreTrainedConfig.from_pretrained(config_path)
        policy_cfg.device = device
        
        # Override pretrained_path to point to weights location
        policy_cfg.pretrained_path = weights_path
        
        # Load dataset metadata from the LIBERO dataset (not the model checkpoint)
        # This provides the feature shapes and normalization stats
        self.get_logger().info("Loading LIBERO dataset metadata...")
        ds_meta = LeRobotDatasetMetadata("Xavier033/pick_place_LIBERO")
        
        # Load custom stats if provided (for your workspace)
        self.action_stats = None
        if stats_path:
            self.get_logger().info(f"Loading action statistics from {stats_path}")
            import json
            from pathlib import Path
            
            # Check if it's a HuggingFace dataset repo ID
            if stats_path.startswith("datasets/") or "/" in stats_path and not Path(stats_path).exists():
                # It's a HF dataset repo - download stats.json from meta/ folder
                from huggingface_hub import hf_hub_download
                self.get_logger().info(f"Downloading meta/stats.json from HuggingFace dataset: {stats_path}")
                stats_file = hf_hub_download(
                    repo_id=stats_path,
                    filename="meta/stats.json",
                    repo_type="dataset"
                )
            else:
                stats_file = stats_path
            
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                # Extract action mean and std
                self.action_stats = {
                    'mean': np.array(stats['action']['mean'], dtype=np.float32),
                    'std': np.array(stats['action']['std'], dtype=np.float32),
                }
                self.get_logger().info(f"Action mean: {self.action_stats['mean']}")
                self.get_logger().info(f"Action std: {self.action_stats['std']}")
        
        # Create policy - it will load weights from pretrained_path in config
        self.policy = make_policy(
            cfg=policy_cfg,
            ds_meta=ds_meta,
        )
        self.policy.eval()
        
        # Get tokenizer - try multiple possible locations
        try:
            # Try processor first (typical for VLMs)
            if hasattr(self.policy.model.vlm_with_expert, 'processor'):
                self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
            # Try language_model
            elif hasattr(self.policy.model.vlm_with_expert, 'language_model'):
                self.tokenizer = self.policy.model.vlm_with_expert.language_model.tokenizer
            # Try text_model
            elif hasattr(self.policy.model.vlm_with_expert, 'text_model'):
                self.tokenizer = self.policy.model.vlm_with_expert.text_model.tokenizer
            else:
                # Fallback: load tokenizer from the VLM model name
                from transformers import AutoTokenizer
                vlm_name = policy_cfg.vlm_model_name  # Should be "HuggingFaceTB/SmolVLM2-500M-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(vlm_name)
                self.get_logger().info(f"Loaded tokenizer from {vlm_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to get tokenizer: {e}")
            raise
            
        self.get_logger().info("✅ Policy loaded successfully")
        
        # Robot Configuration
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # Data buffers
        self.latest_image_top = None
        self.latest_image_side = None
        self.latest_eef_pose = None
        self.latest_joints = None  # Add joint state tracking
        self.data_lock = threading.Lock()
        self.new_data = False
        
        # Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory, 
            "/panda_arm_controller/joint_trajectory", 
            10
        )
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        self.gripper_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        
        # Subscribers
        self.create_subscription(
            CompressedImage, 
            "/camera/rgb/image_raw/compressed", 
            self.top_camera_callback, 
            10
        )
        self.create_subscription(
            CompressedImage, 
            "/camera/gripper/image_raw/compressed", 
            self.side_camera_callback, 
            10
        )
        self.create_subscription(
            Pose, 
            "/actual_end_effector_pose", 
            self.eef_callback, 
            10
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )
        
        # IK Service
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        while not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for IK service...")
        
        # Control rate (10Hz to match training FPS)
        self.control_rate = 10.0
        # Control rate (10Hz to match training FPS)
        self.control_rate = 10.0

        
    # --- CALLBACKS ---
    def top_camera_callback(self, msg):
        with self.data_lock:
            self.latest_image_top = msg
            self.new_data = True
            
    def side_camera_callback(self, msg):
        with self.data_lock:
            self.latest_image_side = msg
            
    def eef_callback(self, msg):
        with self.data_lock:
            self.latest_eef_pose = msg
    
    def joint_callback(self, msg):
        """Track current joint positions for IK seeding"""
        if len(msg.position) >= 7:
            with self.data_lock:
                self.latest_joints = np.array(msg.position[:7], dtype=np.float32)
    
    # --- CONTROL LOGIC ---
    def run_inference_loop(self):
        """Main control loop - runs in separate thread"""
        last_cycle_start = time.time()
        
        while rclpy.ok():
            now = time.time()
            cycle_dt = now - last_cycle_start
            last_cycle_start = now
            
            # Clamp cycle_dt to reasonable bounds for trajectory execution
            # If inference is fast (10Hz), dt is ~0.1s. If slow (0.15Hz), dt is ~6s.
            # We add a small buffer (1.2x) to ensure smooth blending without stopping.
            exec_duration = max(0.5, cycle_dt * 1.2)
            
            with self.data_lock:
                if not self.new_data or any(v is None for v in [
                    self.latest_image_top, 
                    self.latest_image_side, 
                    self.latest_eef_pose,
                    self.latest_joints
                ]):
                    time.sleep(0.01)
                    continue
                
                # Snapshot data
                snap_top = self.latest_image_top
                snap_side = self.latest_image_side
                snap_eef = self.latest_eef_pose
                snap_joints = self.latest_joints
                self.new_data = False
            
            try:
                # 1. Prepare observation
                observation = self.prepare_observation(snap_top, snap_side, snap_eef)
                
                # 2. Get action from policy (new API)
                t0 = time.time()
                with torch.no_grad():
                    action_dict = self.policy.select_action(observation)
                    if isinstance(action_dict, dict):
                        action = action_dict.get("action", action_dict.get("actions"))
                    else:
                        action = action_dict
                inference_time = time.time() - t0
                
                # Log performance occasionally
                self.get_logger().info(
                    f"Inference: {inference_time:.3f}s | Cycle: {cycle_dt:.3f}s | Exec: {exec_duration:.3f}s"
                )
                
                # 3. Execute action (pass adaptive duration)
                self.execute_action(action, snap_eef, snap_joints, duration=exec_duration)
                
            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
            
            # Sleep to maintain rate (only if we are faster than target rate)
            elapsed = time.time() - now
            sleep_time = max(0.0, (1.0 / self.control_rate) - elapsed)
            time.sleep(sleep_time)
    
    def prepare_observation(self, img_top_msg, img_side_msg, eef_pose):
        """Convert ROS messages to policy input format"""
        # Process images: resize to 256x256 and convert to CHW format
        img_top = cv2.resize(
            self.bridge.compressed_imgmsg_to_cv2(img_top_msg, "rgb8"),
            (256, 256)
        ).transpose(2, 0, 1)  # HWC -> CHW
        
        img_side = cv2.resize(
            self.bridge.compressed_imgmsg_to_cv2(img_side_msg, "rgb8"),
            (256, 256)
        ).transpose(2, 0, 1)
        
        # Get current end-effector state
        pos = eef_pose.position
        quat = eef_pose.orientation
        rpy = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        
        # Construct state vector (8-DOF: XYZ, RPY, 2x gripper)
        # Note: We don't have gripper state from your setup, using placeholder
        gripper_state = 1.0  # Assume open for now
        state = np.array([
            pos.x, pos.y, pos.z,
            rpy[0], rpy[1], rpy[2],
            gripper_state, -gripper_state  # Symmetric fingers
        ], dtype=np.float32)
        
        # Tokenize the task instruction
        tokens = self.tokenizer(
            self.task_instruction,
            padding="max_length",
            truncation=True,
            max_length=48,  # Default max length for SmolVLA
            return_tensors="pt"
        )
        
        # Convert to torch tensors and add batch dimension
        # Use dictionary format expected by LeRobot policies
        # IMPORTANT: Normalize images to [0, 1] range (divide by 255)
        '''
            "observation.images.image": torch.from_numpy(img_top).float().unsqueeze(0).to(self.device) / 255.0,
            "observation.images.image2": torch.from_numpy(img_side).float().unsqueeze(0).to(self.device) / 255.0,
        '''
        observation = {
            "observation.images.image": torch.from_numpy(img_top).float().unsqueeze(0).to(self.device),
            "observation.images.image2": torch.from_numpy(img_side).float().unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(self.device),
        }
        
        return observation
    
    def execute_action(self, action, current_eef_pose, current_joints, duration=0.15):
        """Execute predicted action on the robot"""
        # Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = action.cpu().numpy().squeeze()
        
        # Denormalize action if stats are provided
        if self.action_stats is not None:
            # Model outputs normalized actions, convert back to real scale
            action_denorm = action * self.action_stats['std'] + self.action_stats['mean']
            self.get_logger().info(
                f"Action (normalized): dx={action[0]:.4f}, dy={action[1]:.4f}, dz={action[2]:.4f}"
            )
            self.get_logger().info(
                f"Action (denormalized): dx={action_denorm[0]:.4f}, dy={action_denorm[1]:.4f}, dz={action_denorm[2]:.4f}"
            )
            action = action_denorm
        else:
            self.get_logger().info(
                f"Action: dx={action[0]:.4f}, dy={action[1]:.4f}, dz={action[2]:.4f}, "
                f"dr={action[3]:.4f}, dp={action[4]:.4f}, dy={action[5]:.4f}, "
                f"grip={action[6]:.4f}"
            )
        
        # 1. Compute target pose from deltas
        current_pos = current_eef_pose.position
        current_quat = current_eef_pose.orientation
        current_rpy = euler_from_quaternion([
            current_quat.x, current_quat.y, current_quat.z, current_quat.w
        ])
        
        # Apply deltas
        target_x = float(current_pos.x + action[0])
        target_y = float(current_pos.y + action[1])
        target_z = float(current_pos.z + action[2])
        
        target_rpy = [
            float(current_rpy[0] + action[3]),
            float(current_rpy[1] + action[4]),
            float(current_rpy[2] + action[5])
        ]
        target_quat = quaternion_from_euler(*target_rpy)
        
        self.get_logger().info(
            f"Target: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}"
        )
        
        # 2. Control gripper (threshold at 0.0) - convert to Python bool
        gripper_open = bool(action[6] > 0.0)
        self.gripper_pub.publish(Bool(data=gripper_open))
        
        # 3. Publish target pose for visualization
        target_pose = Pose()
        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.position.z = target_z
        target_pose.orientation.x = float(target_quat[0])
        target_pose.orientation.y = float(target_quat[1])
        target_pose.orientation.z = float(target_quat[2])
        target_pose.orientation.w = float(target_quat[3])
        self.target_pub.publish(target_pose)
        
        # 4. Solve IK and execute (seed with current joints!)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = target_pose
        
        joints = self.get_ik_solution(pose_stamped, current_joints)
        if joints:
            self.get_logger().info("✓ IK solved, moving arm")
            self.move_to_joints(joints, duration=duration)  # Use adaptive duration
        else:
            self.get_logger().warn("✗ IK failed - arm not moving")
    
    def get_ik_solution(self, pose_stamped, current_joints):
        """Get IK solution for target pose, seeded with current joints"""
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        # 🌟 CRITICAL: Seed IK with current joint positions
        # This prevents "empty JointState" errors and joint flipping
        if current_joints is not None:
            req.ik_request.robot_state.joint_state.name = self.joint_names
            req.ik_request.robot_state.joint_state.position = current_joints.tolist()
        
        # Create event for async waiting
        event = threading.Event()
        result_wrapper = {"res": None}
        
        def done_callback(future):
            result_wrapper["res"] = future.result()
            event.set()
        
        future = self.ik_client.call_async(req)
        future.add_done_callback(done_callback)
        
        if not event.wait(timeout=3.0):
            self.get_logger().warn("IK service timed out")
            return None
        
        res = result_wrapper["res"]
        if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
            return list(res.solution.joint_state.position[:7])
        else:
            error_code = res.error_code.val if res else "No Response"
            self.get_logger().debug(f"IK failed with error code: {error_code}")
            return None
    
    def move_to_joints(self, joints, duration=0.1):
        """Send joint trajectory command"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = list(joints)
        point.velocities = [0.0] * 7  # Add zero velocities
        point.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        
        traj.points = [point]
        self.arm_pub.publish(traj)
        
        self.get_logger().debug(f"Published joint trajectory: {[f'{j:.3f}' for j in joints]}")
    
    def reset_environment(self):
        """Reset the simulation environment"""
        self.get_logger().info("🔄 Resetting environment...")
        self.pub_aut.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_reset.publish(Bool(data=True))
        time.sleep(4.0)
        self.pub_aut.publish(Bool(data=True))
        time.sleep(0.5)
        self.get_logger().info("✅ Environment reset complete")


def main():
    rclpy.init()
    
    parser = argparse.ArgumentParser(description="SmolVLA Inference for Panda Arm")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="Path or HF repo ID of the model checkpoint"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the blue block",
        help="Task instruction for the policy"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset environment before starting"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to stats.json for action denormalization (from your dataset)"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder in HF repo (e.g., 'checkpoints/003000')"
    )
    
    args = parser.parse_args()
    
    # Create node
    node = SmolVLAInference(
        checkpoint_path=args.checkpoint,
        task_instruction=args.task,
        device=args.device,
        stats_path=args.stats,
        subfolder=args.subfolder,
    )
    
    # Optional reset
    if args.reset:
        node.reset_environment()
    
    node.get_logger().info(f"🎯 Task: {args.task}")
    node.get_logger().info("🚀 Starting inference loop at 10Hz...")
    node.get_logger().info("Press Ctrl+C to stop")
    
    # Run loop in a separate thread so ROS callbacks stay alive
    thread = threading.Thread(target=node.run_inference_loop, daemon=True)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping inference...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()