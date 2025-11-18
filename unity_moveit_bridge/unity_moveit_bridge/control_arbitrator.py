#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose


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

        self.target_pose_pub = self.create_publisher(
            Pose,
            '/target_pose',
            10
        )

        self.autonomous_mode = False
        self.mode_sub = self.create_subscription(
            Bool,
            '/autonomous_mode',
            self.mode_callback,
            10
        )

        self.unity_active = False
        self.unity_timeout = 2.0  # seconds
        self.last_unity_msg_time = None
        
        # Timer to check for Unity timeout
        self.timeout_timer = self.create_timer(0.1, self.check_unity_timeout)

    def mode_callback(self, msg):
        """Enable/disable autonomous mode (ignores Unity during autonomous)"""
        self.autonomous_mode = msg.data
        if msg.data:
            self.get_logger().info('🤖 Autonomous mode enabled - ignoring Unity')
        else:
            self.get_logger().info('👤 Manual mode enabled - Unity can override')

    def bc_pose_callback(self, msg):
        self.bc_pose = msg
        self.get_logger().info('Received BC pose')

        if not self.unity_active:
            self.publish_goal_pose(msg)
            self.get_logger().info("Publishing BC pose")

    def unity_pose_callback(self, msg):
        if self.autonomous_mode:
            return
            
        self.unity_pose = msg
        self.last_unity_msg_time = self.get_clock().now()
        
        self.publish_goal_pose(msg)
        
        if not self.unity_active:
            self.unity_active = True
            self.get_logger().info('Unity control started')

    def publish_goal_pose(self, msg):
        self.target_pose_pub.publish(msg)

    def check_unity_timeout(self):
        """Check if Unity has stopped sending commands"""
        if not self.unity_active or self.last_unity_msg_time is None:
            return
        
        time_since_last = (self.get_clock().now() - self.last_unity_msg_time).nanoseconds / 1e9
        
        if time_since_last > self.unity_timeout:
            self.get_logger().info(f'Unity inactive for {time_since_last:.1f}s - switching to BC')
            self.unity_active = False


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