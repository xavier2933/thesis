import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
import time
import numpy as np

class DumbAlg(Node):
    def __init__(self):
        super().__init__("DumbAlg")

        self.bc_pub = self.create_publisher(Pose, '/bc_target_pose', 10)
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pose_pub = self.create_publisher(Pose, '/target_pose', 10)
        self.wrist_angle_pub = self.create_publisher(Float32, '/wrist_angle', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper_cmd_aut', 10)
        
        self.autonomous_mode_pub = self.create_publisher(Bool, '/autonomous_mode', 10)

        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        
        self.current_joint_velocities = None
        self.arm_joint_indices = None

        # Run sequence after a short delay to ensure publishers are ready
        self.timer = self.create_timer(2.0, self.run_sequence)
        self.has_run = False

    def joint_states_callback(self, msg: JointState):
        """Track current joint velocities."""
        if self.arm_joint_indices is None:
            # Find indices of panda arm joints
            self.arm_joint_indices = []
            for joint_name in ["panda_joint1", "panda_joint2", "panda_joint3",
                              "panda_joint4", "panda_joint5", "panda_joint6",
                              "panda_joint7"]:
                if joint_name in msg.name:
                    self.arm_joint_indices.append(msg.name.index(joint_name))
        
        if len(msg.velocity) > 0:
            self.current_joint_velocities = [msg.velocity[i] for i in self.arm_joint_indices 
                                            if i < len(msg.velocity)]
            # Debug logging
            max_vel = max(abs(v) for v in self.current_joint_velocities)
            if max_vel > 0.001:  # Only log if there's significant movement
                self.get_logger().debug(f"Joint velocities - max: {max_vel:.4f}")

    def is_arm_stopped(self, threshold=0.01):
        """Check if arm has stopped moving."""
        if self.current_joint_velocities is None:
            self.get_logger().debug("⚠️ No joint velocity data yet")
            return False
        max_vel = max(abs(v) for v in self.current_joint_velocities)
        is_stopped = max_vel < threshold
        if not is_stopped:
            self.get_logger().debug(f"Still moving - max velocity: {max_vel:.4f}")
        return is_stopped

    def wait_for_arm_stop(self, timeout=15.0, min_wait=3.0):
        """Wait until arm stops moving or timeout occurs."""
        self.get_logger().info("⏳ Waiting for arm to stop moving...")
        start_time = time.time()
        
        # Always wait at least min_wait seconds
        while rclpy.ok() and (time.time() - start_time) < min_wait:
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # Then check if arm has stopped
        while rclpy.ok() and (time.time() - start_time) < timeout:
            # THIS IS KEY: spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.05)
            
            if self.is_arm_stopped():
                elapsed = time.time() - start_time
                self.get_logger().info(f"✅ Arm has stopped moving (took {elapsed:.2f}s)")
                return True
        
        elapsed = time.time() - start_time
        self.get_logger().warn(f"⚠️ Timeout waiting for arm to stop (waited {elapsed:.2f}s)")
        return False

    def set_autonomous_mode(self, enabled):
        """Enable or disable autonomous mode in the arbitrator."""
        msg = Bool()
        msg.data = enabled
        self.autonomous_mode_pub.publish(msg)
        
        # Spin to ensure message is sent
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

    def move_arm_to_pose(self, x, y, z, wrist_angle=0.0):
        """Move arm to a target pose and wait for completion."""
        # Publish wrist angle
        wrist_msg = Float32()
        wrist_msg.data = wrist_angle
        self.wrist_angle_pub.publish(wrist_msg)
        
        # Publish target pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        
        self.target_pose_pub.publish(pose)
        self.get_logger().info(f"📤 Sent arm to pose: ({x:.3f}, {y:.3f}, {z:.3f}), wrist: {wrist_angle}°")
        
        # Give motion controller time to start trajectory
        self.get_logger().info("⏱️  Waiting for motion to start...")
        start_time = time.time()
        while (time.time() - start_time) < 1.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Wait for arm to finish moving
        return self.wait_for_arm_stop(timeout=15.0)

    def control_gripper(self, open_gripper=True):
        """Open or close the gripper."""
        msg = Bool()
        msg.data = open_gripper
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"🖐️ Gripper {'opening' if open_gripper else 'closing'}...")
        
        # Spin to ensure message is sent and give gripper time to move
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)

    def run_bc(self, duration=5.0):
        """Run behavior cloning for a specified duration."""
        self.get_logger().info("🤖 Starting behavior cloning...")
        
        start_time = time.time()
        while (time.time() - start_time) < duration and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info("✅ Behavior cloning complete")

    def run_sequence(self):
        if self.has_run:
            return
        self.has_run = True
        self.timer.cancel()
        
        try:
            # --- Enable autonomous mode (ignore Unity) ---
            self.get_logger().info("🔒 Enabling autonomous mode...")
            self.set_autonomous_mode(True)

            self.control_gripper(open_gripper=True)

            
            # --- Move arm to position above antenna ---
            self.get_logger().info("📍 Step 1: Moving arm above antenna...")
            if not self.move_arm_to_pose(x=-0.010001915507018566, y=0.2947375476360321, z=0.3815790116786957, wrist_angle=46.0):
                self.get_logger().error("❌ Failed to move arm to starting position")
                return
            
            # Wait 2 seconds between movements
            self.get_logger().info("⏸️  Waiting 2 seconds...")
            start_time = time.time()
            while (time.time() - start_time) < 2.0 and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # self.get_logger().info("📍 Step 2: Rotating wrist...")
            # if not self.move_arm_to_pose(x=-0.010001915507018566, y=0.2947375476360321, z=0.3815790116786957, wrist_angle=46.0):
            #     self.get_logger().error("❌ Failed to rotate wrist")
            #     return
            
            self.get_logger().info("📍 Step 3: Lowering arm...")
            if not self.move_arm_to_pose(x=-0.010001915507018566, y=0.2015794813632965, z=0.3815790116786957, wrist_angle=46.0):
                self.get_logger().error("❌ Failed to lower arm")
                return
            
            self.control_gripper(open_gripper=False)

            self.get_logger().info("📍 Step 3: Lowering arm...")
            if not self.move_arm_to_pose(x=0.3315790891647339, y=0.49157944321632385, z=0.33947375416755676, wrist_angle=46.0):
                self.get_logger().error("❌ Failed to lower arm")
                return

            self.get_logger().info("📍 Step 3: relocation arm...")
            if not self.move_arm_to_pose(x=0.4415774345397949, y=-0.23736822605133057, z=0.38157907128334045, wrist_angle=0.0):
                self.get_logger().error("❌ Failed to lower arm")
                return
            
            self.control_gripper(open_gripper=True)

            self.get_logger().info("📍 Step 3: Lowering arm...")
            if not self.move_arm_to_pose(x=0.3315790891647339, y=0.49157944321632385, z=0.33947375416755676, wrist_angle=46.0):
                self.get_logger().error("❌ Failed to lower arm")
                return
            
            self.get_logger().info("✅ Sequence complete!")
            
            # --- Re-enable Unity control ---
            self.get_logger().info("🔓 Re-enabling Unity control...")
            self.set_autonomous_mode(False)
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in sequence: {e}")
            # Make sure to re-enable Unity even if error occurs
            self.set_autonomous_mode(False)


def main(args=None):
    rclpy.init(args=args)
    node = DumbAlg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()