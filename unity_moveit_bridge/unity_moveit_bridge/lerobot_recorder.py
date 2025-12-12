#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32, Bool
import numpy as np
import pandas as pd
import time
import os

class LerobotRecorder(Node):
    def __init__(self):
        super().__init__('lerobot_recorder')
        
        # --- Configuration ---
        self.frequency = 30.0
        self.output_file = 'dataset.parquet'
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7"
        ]
        
        # --- State Buffers ---
        self.latest_joint_state = None
        self.latest_gripper_state = 0.0 # Assumed 0.0 initially
        self.latest_target_pose = None
        self.latest_wrist_angle = 0.0
        self.latest_gripper_cmd = False
        
        # --- Previous State for Delta Calculation ---
        self.prev_target_pose = None
        self.prev_wrist_angle = None
        self.prev_gripper_cmd_val = None
        
        # --- Data Storage ---
        self.recorded_data = [] # List of dicts
        self.episode_index = 0
        self.frame_index = 0
        self.start_time = time.time()
        
        # --- Subscribers ---
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        # Using Float32 for gripper state as per plan/user indication (verify topic type if error matches)
        self.create_subscription(Float32, '/gripper_state', self.gripper_state_callback, 10) 
        self.create_subscription(Pose, '/target_pose', self.target_pose_callback, 10)
        self.create_subscription(Float32, '/wrist_angle', self.wrist_angle_callback, 10)
        self.create_subscription(Bool, '/gripper_command', self.gripper_cmd_callback, 10)
        
        # --- Timer ---
        self.timer = self.create_timer(1.0 / self.frequency, self.record_step)
        
        self.get_logger().info(f"✅ LeRobot Recorder started. Recording at {self.frequency}Hz to {self.output_file}")

    def joint_state_callback(self, msg):
        # Filter/Sort joints to ensure consistent order if needed, or just store as is if order matches
        # For MoveIt/Panda, usually we want specific 7 joints. 
        # Here we just take the position if names match, or take first 7.
        # Let's map by name to be safe.
        if self.latest_joint_state is None:
             self.latest_joint_state = np.zeros(7)

        name_map = {name: i for i, name in enumerate(msg.name)}
        positions = np.zeros(7)
        found_any = False
        for i, target_name in enumerate(self.joint_names):
            if target_name in name_map:
                positions[i] = msg.position[name_map[target_name]]
                found_any = True
        
        if found_any:
            self.latest_joint_state = positions

    def gripper_state_callback(self, msg):
        self.latest_gripper_state = msg.data

    def target_pose_callback(self, msg):
        self.latest_target_pose = msg

    def wrist_angle_callback(self, msg):
        self.latest_wrist_angle = msg.data

    def gripper_cmd_callback(self, msg):
        self.latest_gripper_cmd = msg.data

    def record_step(self):
        # Check if we have received critical data at least once
        if self.latest_joint_state is None or self.latest_target_pose is None:
            return

        # --- observation.state ---
        # 7 joints + 1 gripper state
        obs_state = np.concatenate([self.latest_joint_state, [self.latest_gripper_state]])
        
        # --- action (deltas) ---
        # [dx, dy, dz, dwrist, dgripper]
        
        # Current command values
        curr_x = self.latest_target_pose.position.x
        curr_y = self.latest_target_pose.position.y
        curr_z = self.latest_target_pose.position.z
        curr_wrist = self.latest_wrist_angle
        # Map boolean False/True to 0.0/0.08 (approx closed/open width or generic command val)
        # Using 0.08 for open based on unity_target_pubsub.py logic (position = 0.08 if open_gripper)
        curr_gripper_val = 0.08 if self.latest_gripper_cmd else 0.0
        
        actions = np.zeros(5)
        
        if self.prev_target_pose is not None:
             actions[0] = curr_x - self.prev_target_pose.position.x
             actions[1] = curr_y - self.prev_target_pose.position.y
             actions[2] = curr_z - self.prev_target_pose.position.z
             actions[3] = curr_wrist - self.prev_wrist_angle
             actions[4] = curr_gripper_val - self.prev_gripper_cmd_val
        
        # Update previous
        self.prev_target_pose = self.latest_target_pose
        self.prev_wrist_angle = curr_wrist
        self.prev_gripper_cmd_val = curr_gripper_val
        
        # Record
        row = {
            'timestamp': time.time() - self.start_time,
            'observation.state': obs_state.astype(np.float32).tolist(), # Store as list for parquet compatibility initially or handle properly
            'action': actions.astype(np.float32).tolist(),
            'episode_index': self.episode_index,
            'frame_index': self.frame_index,
            'index': self.frame_index # Global index if single episode
        }
        self.recorded_data.append(row)
        self.frame_index += 1
        
        if self.frame_index % 100 == 0:
            self.get_logger().info(f"Recorded {self.frame_index} frames...")

    def save_dataset(self):
        if not self.recorded_data:
            self.get_logger().warn("No data recorded to save.")
            return

        self.get_logger().info(f"Saving {len(self.recorded_data)} frames to {self.output_file}...")
        df = pd.DataFrame(self.recorded_data)
        
        # Ensure list columns are handled meaningfully if parquet requires specific types, 
        # but pyarrow usually handles lists of floats fine.
        try:
            df.to_parquet(self.output_file, engine='pyarrow') # or fastparquet
            self.get_logger().info("✅ Dataset saved successfully.")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save dataset: {e}")
            # Fallback to CSV if parquet fails?
            try:
                csv_file = self.output_file.replace('.parquet', '.csv')
                df.to_csv(csv_file)
                self.get_logger().info(f"Saved as CSV instead: {csv_file}")
            except Exception as e2:
                self.get_logger().error(f"Failed to save CSV backup: {e2}")

def main(args=None):
    rclpy.init(args=args)
    node = LerobotRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_dataset()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
