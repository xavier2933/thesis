#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import os
from pathlib import Path




class ControlArbitrator(Node):
    def __init__(self):
        super().__init__("control_arbitrator")
        self.unity_pose = Pose()  
        self.unity_sub = self.create_subscription(
            Pose,
            '/unity_target_pose',
            self.unity_pose_callback,
            10
        )
        
        self.bc_pose = Pose()
        self.bc_sub = self.create_subscription(
            Pose,
            '/bc_target_pose',
            self.bc_pose_callback,
            10
        )
        
        self.is_expert_active = False
        self.expert_active_pub = self.create_publisher(
            Bool,
            '/is_expert_active',
            10
        )

        temp = Bool()
        temp.data=False
        self.expert_active_pub.publish(temp)

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.target_pose_pub = self.create_publisher(
            Pose,
            '/target_pose',
            10
        )
        # For DAgger data collection:
        self.dagger_buffer = []
        self.collection_rate = 10.0  # Hz

        script_dir = Path(__file__).parent
        self.data_dir = script_dir.parent / 'data'
        self.data_dir.mkdir(exist_ok=True)  # Create if doesn't exist
        
        self.get_logger().info(f'DAgger data will be saved to: {self.data_dir}')
        
        # Find next available file number
        self.dagger_session_number = self._get_next_session_number()
        
        # Timer for data collection
        # self.collection_timer = self.create_timer(
        #     1.0 / self.collection_rate,
        #     self.collection_callback
        # )

        self.expert_timeout = 2.0  # seconds
        self.last_unity_msg_time = None
        
        # Timer to check for expert timeout
        self.timeout_timer = self.create_timer(0.1, self.check_expert_timeout)

    def bc_pose_callback(self, msg):
        self.bc_pose = msg
        self.get_logger().info('received bc pose')

        if not self.is_expert_active:
            self.publish_goal_pose(msg)
        else:
            self.get_logger().info("Expert active, passing")
            pass

    def unity_pose_callback(self, msg):
        self.unity_pose = msg
        self.last_unity_msg_time = self.get_clock().now()

        self.publish_goal_pose(msg)
        if not self.is_expert_active:
            self.publish_expert_status(True)
            self.get_logger().info('Expert intervention started - collecting data')


    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.current_joints = list(msg.position[:7])

    def collection_callback(self):
        """Records data at fixed rate when expert is active"""
        if self.is_expert_active and hasattr(self, 'current_joints'):
            # Record current state and expert's action
            sample = {
                'state': self.current_joints.copy(),  # Current joint positions
                'action': self._pose_to_array(self.unity_pose)  # Expert's commanded pose
            }
            self.dagger_buffer.append(sample)
            # self.get_logger().info(f'Collected sample {len(self.dagger_buffer)}')

    def publish_expert_status(self, is_active):
        """Publish expert status and handle data saving"""
        was_active = self.is_expert_active
        self.is_expert_active = is_active
        
        msg = Bool()
        msg.data = is_active
        self.expert_active_pub.publish(msg)
        
        # Save data when expert becomes inactive
        if was_active and not is_active:
            self.get_logger().info('Expert intervention ended - saving data')
            self.save_dagger_data()

    def publish_goal_pose(self, msg):
        self.target_pose_pub.publish(msg)
        self.get_logger().info('Published goal pose to /target_pose')

    def _get_next_session_number(self):
        """Find the next available session number"""
        import glob
        existing_files = glob.glob(str(self.data_dir / 'dagger_session_*.pkl'))
        if not existing_files:
            return 1
        
        # Extract numbers from filenames
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('_')[-1].split('.')[0])
                numbers.append(num)
            except:
                pass
        
        return max(numbers) + 1 if numbers else 1
    
    def _pose_to_array(self, pose):
        """Convert Pose message to numpy array [x, y, z, qx, qy, qz, qw]"""
        return [
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, 
            pose.orientation.z, pose.orientation.w
        ]

    def save_dagger_data(self):
        """Save collected data to pickle file"""
        if len(self.dagger_buffer) == 0:
            self.get_logger().warn('No data to save!')
            return
        
        import pickle
        filename = self.data_dir / f'dagger_session_{self.dagger_session_number:03d}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self.dagger_buffer, f)
        
        self.get_logger().info(f'Saved {len(self.dagger_buffer)} samples to {filename}')
        
        # Increment for next session
        self.dagger_session_number += 1
        # Clear buffer for next collection
        self.dagger_buffer = []

    def check_expert_timeout(self):
        """Check if expert has stopped sending commands"""
        if not self.is_expert_active or self.last_unity_msg_time is None:
            return
        
        time_since_last = (self.get_clock().now() - self.last_unity_msg_time).nanoseconds / 1e9
        
        if time_since_last > self.expert_timeout:
            self.get_logger().info(f'Expert inactive for {time_since_last:.1f}s - stopping intervention')
            self.publish_expert_status(False)


def main(args=None):
    rclpy.init(args=args)
    node = ControlArbitrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    
    