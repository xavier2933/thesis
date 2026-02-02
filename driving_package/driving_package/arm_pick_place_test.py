#!/usr/bin/env python3
"""
Standalone arm pick-and-place test script.
Handles ONLY the arm sequence: reset, spawn, and pick up the block.

This script is useful for tuning the hardcoded offsets when the object changes.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import quaternion_multiply, quaternion_from_euler
import tf2_ros
import numpy as np
import time
import threading


class ArmPickPlaceTest(Node):
    """Standalone node for testing arm pick-and-place with tunable offsets."""
    
    # ==================== TUNABLE OFFSETS ====================
    # Adjust these values when the object (block) changes
    
    # Block pickup offsets (relative to block TF)
    BLOCK_X_OFFSET = -0.04      # X offset from block center
    dy = 398.1286-398.0723

    BLOCK_Y_OFFSET = -0.012 + dy  # Y offset from block center

    dz = 20.12106 - 19.97276 -0.06
    
    # Pickup heights (relative to block Z)
    APPROACH_HEIGHT = 0.13 + dz    # Height for initial approach
    GRASP_HEIGHT = 0.07 + dz      # Height for grasping
    LIFT_HEIGHT = 0.15 + dz       # Height after lifting
    
    # Movement durations (seconds)
    APPROACH_DURATION = 3.0
    GRASP_DURATION = 3.0
    LIFT_DURATION = 2.0
    
    # Gripper orientation correction (Euler angles in radians)
    # This rotates the gripper relative to the block's orientation
    # Original block used: roll=-π/2, pitch=0, yaw=0
    # ORIENTATION_CORRECTION_ROLL = -1.57079632679   # -90 degrees around X
    ORIENTATION_CORRECTION_ROLL = 0   # -90 degrees around X

    ORIENTATION_CORRECTION_PITCH = -1.57079632679              # Rotation around Y
    ORIENTATION_CORRECTION_YAW = 0.0                # Rotation around Z
    
    # Set to True to IGNORE block orientation and use a fixed top-down approach
    USE_FIXED_ORIENTATION = False
    # Fixed orientation (only used if USE_FIXED_ORIENTATION is True)
    # This is a top-down grasp orientation
    FIXED_ORIENTATION_ROLL = 3.14159   # π (pointing down)
    FIXED_ORIENTATION_PITCH = 0.0
    FIXED_ORIENTATION_YAW = 0.0
    
    # =========================================================
    
    def __init__(self):
        super().__init__('arm_pick_place_test')
        
        # --- Arm Control Config ---
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.block_frame = "block"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # IK Client
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        
        # --- Publishers ---
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.gripper_aut_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.spawn_pub = self.create_publisher(Empty, '/spawn_blocc', 10)
        
        # --- Subscribers ---
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        
        # --- State ---
        self.latest_joints = None
        
        self.get_logger().info("ArmPickPlaceTest node initialized")
        self.get_logger().info(f"Block offsets: X={self.BLOCK_X_OFFSET}, Y={self.BLOCK_Y_OFFSET}")
        self.get_logger().info(f"Heights: approach={self.APPROACH_HEIGHT}, grasp={self.GRASP_HEIGHT}, lift={self.LIFT_HEIGHT}")
    
    def joint_callback(self, msg):
        """Store the latest joint states."""
        if len(msg.position) >= 7:
            self.latest_joints = np.array(msg.position[:7], dtype=np.float32)
    
    def refresh_tf(self):
        """Re-initializes TF buffer to handle simulation time resets."""
        self.get_logger().info("🔄 Refreshing TF Buffer...")
        if hasattr(self, 'tf_listener'):
            del self.tf_listener
        if hasattr(self, 'tf_buffer'):
            self.tf_buffer.clear()
            del self.tf_buffer
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Wait a moment for buffer to fill
        time.sleep(1.0)
    
    def reset_environment(self):
        """Reset the simulation environment."""
        self.get_logger().info("🔄 Resetting Environment...")
        self.pub_aut.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_reset.publish(Bool(data=True))
        time.sleep(4.0)
        self.pub_aut.publish(Bool(data=True))
        time.sleep(0.5)
    
    def control_gripper(self, open_gripper: bool):
        """Control the gripper (True=open, False=close)."""
        msg = Bool(data=open_gripper)
        self.gripper_aut_pub.publish(msg)
        time.sleep(1.0)
    
    def get_ik_solution(self, pose: PoseStamped):
        """Get IK solution for a target pose."""
        req = GetPositionIK.Request()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.avoid_collisions = True
        
        # Create an event to block this thread without spinning ROS
        event = threading.Event()
        result_wrapper = {"res": None}
        
        def done_callback(future):
            result_wrapper["res"] = future.result()
            event.set()
        
        future = self.ik_client.call_async(req)
        future.add_done_callback(done_callback)
        
        if not event.wait(timeout=2.0):
            self.get_logger().error("IK service timed out")
            return None
        
        res = result_wrapper["res"]
        if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
            return res.solution.joint_state.position[:7]
        
        self.get_logger().warn(f"IK failed with code: {res.error_code.val if res else 'No Response'}")
        return None
    
    def move_to_pose(self, x, y, z, q, duration):
        """Move arm to a target pose using IK and trajectory execution."""
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        
        joints = self.get_ik_solution(pose)
        if joints:
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            pt = JointTrajectoryPoint(positions=joints, time_from_start=rclpy.duration.Duration(seconds=duration).to_msg())
            traj.points = [pt]
            self.arm_pub.publish(traj)
            time.sleep(duration + 0.5)
        else:
            self.get_logger().error(f"IK Failed for pose ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def pick_up_block(self):
        """Execute the block pickup sequence (reset, approach, grasp, lift)."""
        self.get_logger().info("🤖 Starting block pickup sequence...")
        
        # Step 1: Reset environment
        self.reset_environment()
        
        # Step 2: Spawn block
        self.get_logger().info("📦 Spawning block...")
        self.spawn_pub.publish(Empty())
        time.sleep(1.0)
        
        try:
            # Step 3: Refresh TF buffer to handle simulation time reset
            self.refresh_tf()
            
            # Step 4: Open gripper
            self.get_logger().info("👐 Opening gripper...")
            self.control_gripper(True)
            
            # Step 5: Lookup block transform
            self.get_logger().info("🔍 Looking up block transform...")
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.block_frame, rclpy.time.Time())
            
            # Log the raw block position
            raw_x = tf.transform.translation.x
            raw_y = tf.transform.translation.y
            raw_z = tf.transform.translation.z
            self.get_logger().info(f"📍 Block TF (raw): x={raw_x:.4f}, y={raw_y:.4f}, z={raw_z:.4f}")
            
            # Get block orientation as quaternion
            q_block = [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            ]
            
            # Convert to Euler for logging/debugging
            from tf_transformations import euler_from_quaternion
            block_euler = euler_from_quaternion(q_block)
            self.get_logger().info(f"🔄 Block orientation (quaternion): x={q_block[0]:.4f}, y={q_block[1]:.4f}, z={q_block[2]:.4f}, w={q_block[3]:.4f}")
            self.get_logger().info(f"🔄 Block orientation (Euler rad): roll={block_euler[0]:.4f}, pitch={block_euler[1]:.4f}, yaw={block_euler[2]:.4f}")
            self.get_logger().info(f"🔄 Block orientation (Euler deg): roll={np.degrees(block_euler[0]):.1f}°, pitch={np.degrees(block_euler[1]):.1f}°, yaw={np.degrees(block_euler[2]):.1f}°")
            
            # Determine gripper orientation
            if self.USE_FIXED_ORIENTATION:
                # Use a fixed orientation (ignores block orientation)
                self.get_logger().info("📐 Using FIXED orientation (ignoring block TF rotation)")
                q_fixed = quaternion_from_euler(
                    self.FIXED_ORIENTATION_ROLL,
                    self.FIXED_ORIENTATION_PITCH,
                    self.FIXED_ORIENTATION_YAW
                )
            else:
                # Apply orientation correction relative to block
                self.get_logger().info(f"📐 Applying orientation correction: roll={self.ORIENTATION_CORRECTION_ROLL:.4f}, pitch={self.ORIENTATION_CORRECTION_PITCH:.4f}, yaw={self.ORIENTATION_CORRECTION_YAW:.4f}")
                q_correction = quaternion_from_euler(
                    self.ORIENTATION_CORRECTION_ROLL,
                    self.ORIENTATION_CORRECTION_PITCH,
                    self.ORIENTATION_CORRECTION_YAW
                )
                q_fixed = quaternion_multiply(q_block, q_correction)
            
            # Log final gripper orientation
            gripper_euler = euler_from_quaternion(q_fixed)
            self.get_logger().info(f"✅ Final gripper orientation (Euler deg): roll={np.degrees(gripper_euler[0]):.1f}°, pitch={np.degrees(gripper_euler[1]):.1f}°, yaw={np.degrees(gripper_euler[2]):.1f}°")
            
            # Get block position with offsets
            tx = raw_x + self.BLOCK_X_OFFSET
            ty = raw_y + self.BLOCK_Y_OFFSET
            tz = raw_z
            self.get_logger().info(f"📍 Block TF (with offsets): x={tx:.4f}, y={ty:.4f}, z={tz:.4f}")
            
            # Step 6: Move to approach position (above block)
            self.get_logger().info(f"📍 Moving to approach position (z + {self.APPROACH_HEIGHT})...")
            self.move_to_pose(tx, ty, tz + self.APPROACH_HEIGHT, q_fixed, self.APPROACH_DURATION)
            
            # Step 7: Move down to grasp position
            self.get_logger().info(f"📍 Moving down to grasp position (z + {self.GRASP_HEIGHT})...")
            self.move_to_pose(tx, ty, tz + self.GRASP_HEIGHT, q_fixed, self.GRASP_DURATION)
            
            # Step 8: Close gripper
            self.get_logger().info("✊ Closing gripper...")
            self.control_gripper(False)
            
            # Step 9: Lift block
            self.get_logger().info(f"📍 Lifting block (z + {self.LIFT_HEIGHT})...")
            self.move_to_pose(tx, ty, tz + self.LIFT_HEIGHT, q_fixed, self.LIFT_DURATION)
            
            self.get_logger().info("✅ Block pickup complete!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Pickup error: {e}")
            return False
    
    def run_test(self):
        """Run the pick test once and shutdown."""
        self.get_logger().info("\n" + "=" * 50)
        self.get_logger().info("🚀 ARM PICK-PLACE TEST")
        self.get_logger().info("=" * 50)
        self.get_logger().info("This script will:")
        self.get_logger().info("  1. Reset the environment")
        self.get_logger().info("  2. Spawn the block")
        self.get_logger().info("  3. Pick up the block")
        self.get_logger().info("=" * 50 + "\n")
        
        # Wait for IK service
        self.get_logger().info("⏳ Waiting for IK service...")
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ IK service not available!")
            return
        self.get_logger().info("✅ IK service available")
        
        # Give time for other nodes to be ready
        time.sleep(2.0)
        
        # Execute pickup
        success = self.pick_up_block()
        
        if success:
            self.get_logger().info("\n🎉 TEST PASSED - Block picked up successfully!")
        else:
            self.get_logger().error("\n❌ TEST FAILED - Check offsets and retry")
        
        self.get_logger().info("\nTest complete. Node will keep running for inspection.")
        self.get_logger().info("Press Ctrl+C to exit.\n")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ArmPickPlaceTest()
        
        # Run the test in a separate thread so we can spin
        test_thread = threading.Thread(target=node.run_test, daemon=True)
        test_thread.start()
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
