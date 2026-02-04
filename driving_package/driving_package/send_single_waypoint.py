#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import math
import time

class SingleWaypointSender(Node):
    def __init__(self):
        super().__init__('single_waypoint_sender')
        
        # This matches the 'waypointTopicName' in your Unity RoverROSComms script
        self.publisher = self.create_publisher(Pose, 'rover/waypoint', 10)
        
        self.get_logger().info("Ready to send waypoint with heading...")

    def euler_to_quaternion(self, yaw_degrees):
        """Converts yaw (degrees) to a ROS Quaternion."""
        yaw = math.radians(yaw_degrees)
        # For a simple 2D rotation (Yaw), only Z and W are used
        # 
        return {
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(yaw / 2),
            'w': math.cos(yaw / 2)
        }

    def send_waypoint(self, x, y, z, heading_deg):
        msg = Pose()
        
        # Position mapping
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        
        # Orientation mapping
        q = self.euler_to_quaternion(heading_deg)
        msg.orientation.x = q['x']
        msg.orientation.y = q['y']
        msg.orientation.z = q['z']
        msg.orientation.w = q['w']
        
        self.get_logger().info(f"Sending Goal: Pos({x}, {y}, {z}) | Heading: {heading_deg}°")
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    sender = SingleWaypointSender()

    # Allow time for ROS discovery (connections to establish)
    time.sleep(1.0)

    # --- COMMANDING THE ROVER ---
    # Position: 410, 18, 230 | Orientation: 90 degrees
    sender.send_waypoint(x=410.0, y=18.0, z=230.0, heading_deg=90.0)

    # Keep alive for a moment to ensure message is sent
    time.sleep(1.0)
    
    sender.get_logger().info("Mission dispatched. Shutting down sender.")
    sender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()