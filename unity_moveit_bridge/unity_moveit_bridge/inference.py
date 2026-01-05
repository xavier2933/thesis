#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import argparse
import threading
import time
import torch
from pathlib import Path

# ROS2 Messages
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes

# Utils
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
import tf2_ros

# LeRobot
# LeRobot
# LeRobot
from lerobot.policies.factory import make_policy, get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from huggingface_hub import snapshot_download

class SmolVLAInference(Node):
    def __init__(self, repo_id, dataset_repo, fps=10, device="cuda", revision=None, subfolder=None):
        super().__init__("smol_vla_inference")
        self.bridge = CvBridge()
        self.fps = fps
        self.device = device
        
        # --- 1. Load Model ---
        self.get_logger().info(f"📥 Loading model: {repo_id} (Revision: {revision}, Subfolder: {subfolder})")
        
        # If subfolder is provided, download it first and update repo_id to local path
        if subfolder:
            self.get_logger().info(f"📂 Downloading subfolder '{subfolder}' from {repo_id}...")
            repo_path = snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=f"{subfolder}/*")
            # Update repo_id to point to the local subfolder
            repo_id = str(Path(repo_path) / subfolder)
            self.get_logger().info(f"✅ Downloaded to: {repo_id}")
            # Clear revision/subfolder for subsequent calls since we are now pointing to local path
            revision = None 
            subfolder = None

        # Load config first
        cfg = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
        cfg.pretrained_path = repo_id 
        cfg.device = device
        
        # Load Metadata from dataset (for normalization stats)
        self.get_logger().info(f"📊 Loading dataset stats from: {dataset_repo}")
        
        # Try to locate the dataset root in the default cache
        dataset_root = Path.home() / ".cache/huggingface/lerobot" / dataset_repo
        
        if dataset_root.exists():
            ds_meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
        else:
            self.get_logger().warn(f"Dataset not found at {dataset_root}. Attempting to use repo_id directly...")
            ds_meta = LeRobotDatasetMetadata(dataset_repo)

        self.ds_meta = ds_meta

        if revision:
            # Manual instantiation to support revision (only if still using remote repo)
            self.get_logger().info(f"🛠️  Using manual policy loading for revision: {revision}")
            policy_cls = get_policy_class(cfg.type)
            features = dataset_to_policy_features(ds_meta.features)
            
            # Setup features in config
            cfg.output_features = {k:v for k,v in features.items() if v.type is FeatureType.ACTION}
            if not cfg.input_features:
                cfg.input_features = {k:v for k,v in features.items() if k not in cfg.output_features}
            
            self.policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg, revision=revision)
        else:
            # Standard loading (also works for local paths from subfolder download)
            self.policy = make_policy(cfg, ds_meta=ds_meta)
            
        self.policy.eval()
        
        # Create Pre/Post Processors (official way to handle normalization)
        self.get_logger().info("🔄 Creating Pre/Post Processors...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.policy.config.pretrained_path,
            dataset_stats=self.ds_meta.stats
        )
        
        self.get_logger().info("✅ Model loaded successfully")

        # --- 2. State & Buffers ---
# ... (rest of class)

# ... (main function)
def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace model repo ID")
    parser.add_argument("--dataset", type=str, default="Xavier033/test_pick_place", help="Dataset repo ID for stats")
    parser.add_argument("--revision", type=str, default=None, help="Model checkpoint revision (git branch/tag)")
    parser.add_argument("--subfolder", type=str, default=None, help="Model subfolder (e.g. checkpoints/005000)")
    parser.add_argument("--fps", type=int, default=10, help="Inference FPS")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    node = SmolVLAInference(
        repo_id=args.repo, 
        dataset_repo=args.dataset, 
        fps=args.fps, 
        device=args.device, 
        revision=args.revision,
        subfolder=args.subfolder
    )
    
    # Run inference logic in separate thread
    thread = threading.Thread(target=node.run_inference, daemon=True)
    self.latest_image_top = None
    self.latest_image_side = None
    self.latest_joints = None
    self.latest_eef_pose = None
    self.latest_gripper_state = 0.0 # From joint states or assumed
    self.data_lock = threading.Lock()
    
    self.base_frame = "panda_link0"
    self.ee_link = "panda_hand"
    self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]

    # --- 3. ROS Setup ---
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    # Service Client for IK
    self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
    while not self.ik_client.wait_for_service(timeout_sec=1.0):
        self.get_logger().info("⏳ Waiting for IK service...")

    # Publishers
    self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
    self.gripper_aut_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
    self.target_vis_pub = self.create_publisher(Pose, "/inference_target_pose", 10)

    # Subscribers (Matching orchestrate_data_collection.py)
    self.create_subscription(CompressedImage, "/camera/rgb/image_raw/compressed", self.top_camera_callback, 10)
    self.create_subscription(CompressedImage, "/camera/gripper/image_raw/compressed", self.side_camera_callback, 10)
    self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
    self.create_subscription(Pose, "/actual_end_effector_pose", self.eef_callback, 10)

    # Control Loop
    self.timer = self.create_timer(1.0 / self.fps, self.control_loop)
    
    # Fixed instructions for now (could be an argument)
    self.instruction = "Pick up the blue block"
    self.get_logger().info(f"👉 Active Instruction: {self.instruction}")

    # --- Callbacks ---
    def top_camera_callback(self, msg):
        with self.data_lock: self.latest_image_top = msg
    
    def side_camera_callback(self, msg):
        with self.data_lock: self.latest_image_side = msg

    def joint_callback(self, msg):
        # Assumes panda joints are first or named (standard panda mapping)
        # Orchestrate simply takes first 7
        if len(msg.position) >= 7:
            with self.data_lock: 
                self.latest_joints = np.array(msg.position[:7], dtype=np.float32)

    def eef_callback(self, msg):
        with self.data_lock: self.latest_eef_pose = msg

    # --- IK Helper ---
    def get_ik_solution(self, pose: PoseStamped, start_joints=None):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        if start_joints is not None:
             rs = MoveItErrorCodes() # dummy placeholder if needed, but we need RobotState
             # Construct minimal robot state if needed, or rely on moveit current state
             # For speed, we often let moveit use current state, but providing it is better
             pass

        # Call service synchronously for simplicity in the control thread
        try:
            future = self.ik_client.call_async(req)
            # We are in a timer callback, so we can't spin until future complete easily
            # But we can wait on the future since we are in a separate thread/timer context?
            # Actually rclpy timers run in the executor. Blocking here is bad if single threaded.
            # But we need the result to proceed.
            # Best practice: use a separate thread for the control loop or use synchronous call if possible?
            # rclpy doesn't support sync calls well inside async callbacks.
            # However, orchestrate used a clever threading.Event trick. I will assume we can't block too long.
            # For inference, maybe we should run the logic in a separate thread loop instead of a timer.
            pass
        except:
            pass
        return None 
    
    # We will use the 'separate thread loop' pattern from orchestrate.py to allow blocking service calls
    # So 'control_loop' will physically be called by a thread, not a ROS timer.

    def compute_ik_sync(self, target_pose, current_joints):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = target_pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        # 🌟 CRITICAL FIX: Seed IK with current joints to prevent "flipping" 
        # and silence "empty JointState" errors.
        if current_joints is not None:
             req.ik_request.robot_state.joint_state.name = self.joint_names
             req.ik_request.robot_state.joint_state.position = current_joints.tolist()

        # Create an event to await response
        event = threading.Event()
        result = {"response": None}

        def done_cb(future):
            result["response"] = future.result()
            event.set()
        
        future = self.ik_client.call_async(req)
        future.add_done_callback(done_cb)
        
        if not event.wait(timeout=1.0):
            self.get_logger().warn("IK Timeout")
            return None
            
        res = result["response"]
        if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
            return res.solution.joint_state.position[:7]
        return None

    # --- Main Loop ---
    def run_inference(self):
        rate = 1.0 / self.fps
        while rclpy.ok():
            start_time = time.time()
            
            # 1. Gather Observation
            obs_dict = {}
            with self.data_lock:
                if any(x is None for x in [self.latest_image_top, self.latest_image_side, self.latest_eef_pose, self.latest_joints]):
                    # self.get_logger().warn("Waiting for observations...", tuple_once=True)
                    time.sleep(1.0)
                    continue
                
                # Copy data
                snap_top = self.latest_image_top
                snap_side = self.latest_image_side
                snap_eef = self.latest_eef_pose
                snap_joints = self.latest_joints
            
            # 2. Process Inputs
            try:
                # Images
                img_top = self.bridge.compressed_imgmsg_to_cv2(snap_top, "rgb8")
                img_top = cv2.resize(img_top, (224, 224)) # default smol/libero size
                
                img_side = self.bridge.compressed_imgmsg_to_cv2(snap_side, "rgb8")
                img_side = cv2.resize(img_side, (224, 224))
                
                # Convert to Torch Tensors and permute to (C, H, W) and float (0-1 range)
                img_top = torch.from_numpy(img_top).float().permute(2, 0, 1) / 255.0
                img_side = torch.from_numpy(img_side).float().permute(2, 0, 1) / 255.0

                # State: [x,y,z,qx,qy,qz,qw, gripper]
                # Assuming gripper is open=1, closed=-1. 
                # We don't have direct gripper feedback in 'orchestrate' other than command.
                # We will approximate from the last commanded or default.
                current_gripper = 1.0 
                
                state_vec = torch.tensor([
                    snap_eef.position.x, snap_eef.position.y, snap_eef.position.z,
                    snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w,
                    current_gripper
                ], dtype=torch.float32)

                joint_vec = torch.tensor(snap_joints, dtype=torch.float32)

                # Prepare Tokenized Text
                # We access the tokenizer from the internal VLM model 
                # (policy.model.vlm_with_expert.processor.tokenizer usually, or usage of processor directly)
                if not hasattr(self, "cached_lang_tokens"):
                    # Processor is usually at self.policy.model.vlm_with_expert.processor
                    processor = self.policy.model.vlm_with_expert.processor
                    # Must pass text as keyword argument, otherwise it might interpret list of strings as image paths
                    text_out = processor(text=[self.instruction], return_tensors="pt", padding=True, truncation=True)
                    self.cached_lang_tokens = text_out["input_ids"]
                    self.cached_lang_mask = text_out["attention_mask"]
                
                # Assemble Raw Observation Dictionary matching validation keys
                observation = {
                    "observation.images.agentview_image": img_top,
                    "observation.images.eye_in_hand_image": img_side,
                    "observation.state": state_vec,
                    "observation.state.joint": joint_vec,
                }
                
                # Move to device and add batch dim BEFORE preprocessor
                for k, v in observation.items():
                    observation[k] = v.to(self.device).unsqueeze(0)

                # Add Text manually (preprocessor usually handles tensors)
                observation["task"] = [self.instruction]
                observation["observation.language.tokens"] = self.cached_lang_tokens.to(self.device)
                observation["observation.language.attention_mask"] = self.cached_lang_mask.to(self.device)

                # 🌟 APPLY PREPROCESSOR (handling normalization)
                observation = self.preprocessor(observation)

            except Exception as e:
                self.get_logger().error(f"Preprocessing Error: {e}")
                continue

            # 3. Inference
            with torch.inference_mode():
                action = self.policy.select_action(observation)
                
                # 🌟 APPLY POSTPROCESSOR (handling unnormalization)
                action = self.postprocessor(action)
                
                action = action.squeeze(0).cpu().numpy()
                
                # 🛑 SAFETY CLAMPING & SCALING 🛑
                # Now that we have properly unnormalized actions, apply safety limits.
                # Keeping the clamps but maybe relaxing scaling if the model behaves better.
                ACTION_SCALE = 0.5  # Reduce magnitude by half
                
                # Apply scale
                action[:6] *= ACTION_SCALE
                
                # Clamp Position Deltas (max 3cm per step)
                MAX_POS_DELTA = 0.03
                action[0] = np.clip(action[0], -MAX_POS_DELTA, MAX_POS_DELTA)
                action[1] = np.clip(action[1], -MAX_POS_DELTA, MAX_POS_DELTA)
                action[2] = np.clip(action[2], -MAX_POS_DELTA, MAX_POS_DELTA)
                
                # Clamp Rotation Deltas (max 0.1 rad per step)
                MAX_ROT_DELTA = 0.1
                action[3] = np.clip(action[3], -MAX_ROT_DELTA, MAX_ROT_DELTA)
                action[4] = np.clip(action[4], -MAX_ROT_DELTA, MAX_ROT_DELTA)
                action[5] = np.clip(action[5], -MAX_ROT_DELTA, MAX_ROT_DELTA)

                self.get_logger().info(f"Act (Safe): {np.round(action, 4)}")

            # 4. Execute Action
            # Extract Delta
            dx, dy, dz = action[0], action[1], action[2]
            dr, dp, dyaw = action[3], action[4], action[5]
            gripper_cmd = action[6]

            # Calculate Target Pose
            # Position
            tx = snap_eef.position.x + dx
            ty = snap_eef.position.y + dy
            tz = snap_eef.position.z + dz

            # Orientation
            # Convert current to RPY
            c_msg_q = [snap_eef.orientation.x, snap_eef.orientation.y, snap_eef.orientation.z, snap_eef.orientation.w]
            c_rpy = euler_from_quaternion(c_msg_q)
            
            # Add deltas
            t_rpy = [c + d for c, d in zip(c_rpy, [dr, dp, dyaw])]
            
            # Convert back to Quat
            t_q = quaternion_from_euler(t_rpy[0], t_rpy[1], t_rpy[2])

            # Prepare Pose Message
            target_pose = PoseStamped()
            target_pose.header.frame_id = self.base_frame
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.pose.position.x = tx
            target_pose.pose.position.y = ty
            target_pose.pose.position.z = tz
            target_pose.pose.orientation.x = t_q[0]
            target_pose.pose.orientation.y = t_q[1]
            target_pose.pose.orientation.z = t_q[2]
            target_pose.pose.orientation.w = t_q[3]

            self.target_vis_pub.publish(target_pose.pose)

            # Solve IK
            joint_sol = self.compute_ik_sync(target_pose, snap_joints)
            
            # Send Command
            if joint_sol:
                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                pt = JointTrajectoryPoint()
                pt.positions = joint_sol
                # Duration should match inference rate roughly, slightly larger for smoothness
                pt.time_from_start = rclpy.duration.Duration(seconds=1.5 / self.fps).to_msg()
                traj.points = [pt]
                self.arm_pub.publish(traj)
            else:
                self.get_logger().warn("IK Failed for inference target")

            # Gripper
            # >= 0 is Open (1.0), < 0 is Closed (-1.0) usually
            open_grip = True if gripper_cmd > 0 else False
            self.gripper_aut_pub.publish(Bool(data=open_grip))

            # Maintain Rate
            elapsed = time.time() - start_time
            sleep_time = max(0, rate - elapsed)
            time.sleep(sleep_time)

    # Note: control_loop is unused because run_inference is threaded, but kept for signature ref if needed.
    def control_loop(self):
        pass

def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace model repo ID")
    parser.add_argument("--dataset", type=str, default="Xavier033/test_pick_place", help="Dataset repo ID for stats")
    parser.add_argument("--revision", type=str, default=None, help="Model checkpoint revision (e.g. checkpoint-5000)")
    parser.add_argument("--fps", type=int, default=10, help="Inference FPS")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    node = SmolVLAInference(repo_id=args.repo, dataset_repo=args.dataset, fps=args.fps, device=args.device, revision=args.revision)
    
    # Run inference logic in separate thread
    thread = threading.Thread(target=node.run_inference, daemon=True)
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
