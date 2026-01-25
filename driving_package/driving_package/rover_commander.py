#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from geometry_msgs.msg import PoseArray
import time
import threading

class RoverCommander(Node):
    def __init__(self):
        super().__init__('rover_commander')
        
        # Publishers
        self.move_cmd_pub = self.create_publisher(Int32, 'rover/move_command', 10)
        
        # Subscribers
        self.plate_sub = self.create_subscription(
            PoseArray,
            'rover/plate_locations',
            self.plate_locations_callback,
            10
        )
        self.status_sub = self.create_subscription(
            String,
            'rover/status',
            self.status_callback,
            10
        )
        
        # State
        self.plate_positions = []
        self.current_plate_index = -1
        self.is_at_plate = False
        self.is_navigating = False
        self.rover_position = [0.0, 0.0, 0.0]
        
        self.get_logger().info("[ROVER] Commander node initialized")
        self.get_logger().info("[ROVER] Waiting for plate locations from Unity...")
        
    def plate_locations_callback(self, msg):
        """Receive plate locations from Unity (sent once at startup)"""
        self.plate_positions = []
        for pose in msg.poses:
            pos = [pose.position.x, pose.position.y, pose.position.z]
            self.plate_positions.append(pos)
        
        self.get_logger().info(f"[ROVER] ✓ Received {len(self.plate_positions)} plate locations:")
        for i, pos in enumerate(self.plate_positions):
            self.get_logger().info(f"  Plate {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    def status_callback(self, msg):
        """Receive rover status updates from Unity"""
        # Format: "plateIndex,isAtPlate,isNavigating,x,y,z"
        try:
            parts = msg.data.split(',')
            prev_plate = self.current_plate_index
            prev_at_plate = self.is_at_plate
            
            self.current_plate_index = int(parts[0])
            self.is_at_plate = parts[1] == 'True'
            self.is_navigating = parts[2] == 'True'
            self.rover_position = [float(parts[3]), float(parts[4]), float(parts[5])]
            
            # Log when we arrive at a plate
            if self.is_at_plate and not prev_at_plate:
                self.get_logger().info(f"[ROVER] ✓ Arrived at plate {self.current_plate_index}")
                    
        except Exception as e:
            self.get_logger().warn(f"[ROVER] Failed to parse status: {e}")
    
    def send_move_command(self, plate_index):
        """
        Command rover to move to a specific plate (non-blocking)
        
        Args:
            plate_index (int): Index of plate to navigate to (0-3)
        """
        if plate_index < 0 or plate_index >= len(self.plate_positions):
            self.get_logger().error(f"[ROVER] Invalid plate index: {plate_index} (valid: 0-{len(self.plate_positions)-1})")
            return
        
        self.get_logger().info(f"[ROVER] → Commanding rover to plate {plate_index}")
        
        # Send command
        msg = Int32()
        msg.data = plate_index
        self.move_cmd_pub.publish(msg)
    
    def get_status(self):
        """Get current rover status"""
        return {
            'current_plate': self.current_plate_index,
            'at_plate': self.is_at_plate,
            'navigating': self.is_navigating,
            'position': self.rover_position
        }


def main(args=None):
    rclpy.init(args=args)
    
    try:
        commander = RoverCommander()
        
        commander.get_logger().info("\n========================================")
        commander.get_logger().info("=== ROVER COMMANDER - MANUAL MODE ===")
        commander.get_logger().info("========================================")
        commander.get_logger().info("Listening for commands on /rover/move_command")
        commander.get_logger().info("\nTo send rover to a plate, use:")
        commander.get_logger().info("  ros2 topic pub --once /rover/move_command std_msgs/msg/Int32 '{data: 0}'")
        commander.get_logger().info("\nWaiting for commands...\n")
        
        rclpy.spin(commander)
        
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()