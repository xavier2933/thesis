#!/usr/bin/env python3
"""
Standalone arm pick-and-place test script.
Handles ONLY the arm sequence: reset, spawn, and pick up the block.

This script is useful for tuning the hardcoded offsets when the object changes.
It mirrors the logic in rover_commander.py exactly, so tuned values can be
copied directly to that class.

Usage:
    ros2 run driving_package arm_pick_place_test

    OR directly:
    python3 arm_pick_place_test.py
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion
import tf2_ros
import numpy as np
import time
import threading


class ArmPickPlaceTest(Node):
    """Standalone node for testing arm pick-and-place with tunable offsets.

    Mirrors rover_commander.RoverCommander's arm methods 1:1 so that any
    tuned values can be pasted directly into that class.
    """

    # ==================== TUNABLE OFFSETS ====================
    # Adjust these values when the object (block) changes.
    # These must match rover_commander.py to have the same behaviour
    # during the real BT deployment.

    # Block pickup offsets (relative to block TF)
    BLOCK_X_OFFSET = -0.04      # X offset from block center
    _dy = 398.1286 - 398.0723
    BLOCK_Y_OFFSET = -0.012 + _dy  # Y offset from block center

    _dz = 20.12106 - 19.97276 - 0.06

    # Pickup heights (relative to block Z).
    # NOTE: +0.02 correction matches rover_commander.py
    APPROACH_HEIGHT = 0.13 + _dz + 0.02   # Height for initial approach
    GRASP_HEIGHT    = 0.07 + _dz + 0.015   # Height for grasping
    LIFT_HEIGHT     = 0.15 + _dz          # Height after lifting

    # Movement durations (seconds)
    APPROACH_DURATION = 3.0
    GRASP_DURATION    = 3.0
    LIFT_DURATION     = 2.0

    # Gripper orientation correction (Euler angles in radians)
    # Applied as: q_final = q_block ⊗ q_correction
    ORIENTATION_CORRECTION_ROLL  = 0.0               # Rotation around X
    ORIENTATION_CORRECTION_PITCH = -1.57079632679    # -90 degrees around Y
    ORIENTATION_CORRECTION_YAW   = 0.0               # Rotation around Z

    # Set to True to IGNORE block orientation and use a fixed top-down approach
    USE_FIXED_ORIENTATION = False
    FIXED_ORIENTATION_ROLL  = 3.14159   # π  (pointing down)
    FIXED_ORIENTATION_PITCH = 0.0
    FIXED_ORIENTATION_YAW   = 0.0

    # ==================== TUNABLE PLACEMENT POSE ==========================
    # Hardcoded placement target in panda_link0 frame.
    # To re-tune: move arm to desired placement position manually, then:
    #   ros2 topic echo /actual_end_effector_pose --once
    # and paste the x/y/z values below.
    PLACE_X = 0.3646       # metres in panda_link0 frame
    PLACE_Y = -0.1692      # metres in panda_link0 frame
    PLACE_Z = -0.3087      # metres in panda_link0 frame (surface of plate)

    # Hover gap above PLACE_Z before lowering
    PLACE_HOVER_OFFSET   = 0.55   # metres
    # Drop gap: how far above plate to release the block
    PLACE_DROP_OFFSET    = 0.04   # metres
    # Retract gap: lift back up after releasing
    PLACE_RETRACT_OFFSET = 0.40   # metres

    # Placement orientation Euler angles (rad)
    # π roll + π/2 yaw = gripper flipped top-down, rotated 90° — matches rover_commander
    PLACE_ROLL  = 3.14159   # π  — flip gripper to point down
    PLACE_PITCH = 0.0
    PLACE_YAW   = 1.5707    # π/2 — rotate 90° around Z
    # ======================================================================

    def __init__(self):
        super().__init__('arm_pick_place_test')

        # --- Arm Control Config ---
        self.base_frame  = "panda_link0"
        self.ee_link     = "panda_hand"
        self.block_frame = "block"
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]

        # TF Buffer and Listener
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # IK Client
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        # --- Publishers ---
        self.pub_reset       = self.create_publisher(Bool,           "/reset_env",                              10)
        self.pub_aut         = self.create_publisher(Bool,           "/autonomous_mode",                        10)
        self.arm_pub         = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.gripper_aut_pub = self.create_publisher(Bool,           "/gripper_cmd_aut",                        10)
        self.spawn_pub       = self.create_publisher(Empty,          '/spawn_blocc',                            10)

        # --- Subscribers ---
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)

        # --- State ---
        self.latest_joints = None

        self.get_logger().info("ArmPickPlaceTest node initialized")
        self.get_logger().info(f"Block offsets: X={self.BLOCK_X_OFFSET:.4f}, Y={self.BLOCK_Y_OFFSET:.4f}")
        self.get_logger().info(
            f"Heights: approach={self.APPROACH_HEIGHT:.4f}, "
            f"grasp={self.GRASP_HEIGHT:.4f}, lift={self.LIFT_HEIGHT:.4f}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def joint_callback(self, msg):
        """Store the latest joint states for IK seeding."""
        if len(msg.position) >= 7:
            self.latest_joints = np.array(msg.position[:7], dtype=np.float64)

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------

    def refresh_tf(self):
        """Re-initialise TF buffer to handle simulation time resets."""
        self.get_logger().info("🔄 Refreshing TF Buffer...")
        if hasattr(self, 'tf_listener'):
            del self.tf_listener
        if hasattr(self, 'tf_buffer'):
            self.tf_buffer.clear()
            del self.tf_buffer

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Give the buffer time to fill
        time.sleep(1.5)

    def lookup_transform_with_retry(self, target_frame, source_frame, retries=10, delay=0.5):
        """Lookup a TF transform, retrying on failure.

        After refresh_tf() the buffer can still be empty for a moment.
        This loop avoids an immediate crash from a single failed lookup.
        """
        for attempt in range(1, retries + 1):
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time()
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(
                    f"TF lookup attempt {attempt}/{retries} failed: {e}"
                )
                if attempt < retries:
                    time.sleep(delay)
        raise RuntimeError(
            f"Could not look up TF {source_frame} → {target_frame} "
            f"after {retries} attempts"
        )

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

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
        """Control the gripper (True=open, False=close).

        The /gripper_cmd_aut topic is only forwarded by unity_target_pubsub.py
        when autonomous_mode is True — which reset_environment() sets at the end.
        """
        msg = Bool(data=open_gripper)
        self.gripper_aut_pub.publish(msg)
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # IK + trajectory helpers
    # ------------------------------------------------------------------

    def get_ik_solution(self, pose: PoseStamped):
        """Get IK solution for a target pose.

        Seeds the IK with the current joint positions to prevent elbow flips.
        This matches rover_commander.get_ik_solution() exactly.
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name     = "panda_arm"
        req.ik_request.pose_stamped   = pose
        req.ik_request.ik_link_name   = self.ee_link
        req.ik_request.avoid_collisions = True

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

        self.get_logger().warn(
            f"IK failed with code: {res.error_code.val if res else 'No Response'}"
        )
        return None

    def move_to_pose(self, x, y, z, q, duration):
        """Move arm to a target pose using IK and trajectory execution."""
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        pose.pose.orientation.x, pose.pose.orientation.y, \
            pose.pose.orientation.z, pose.pose.orientation.w = q

        joints = self.get_ik_solution(pose)
        if joints:
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            pt = JointTrajectoryPoint(
                positions=joints,
                time_from_start=rclpy.duration.Duration(seconds=duration).to_msg()
            )
            traj.points = [pt]
            self.arm_pub.publish(traj)
            time.sleep(duration + 0.5)
        else:
            self.get_logger().error(f"IK Failed for pose ({x:.3f}, {y:.3f}, {z:.3f})")

    # ------------------------------------------------------------------
    # Core pick sequence
    # ------------------------------------------------------------------

    def pick_up_block(self):
        """Execute the block pickup sequence (reset → spawn → approach → grasp → lift).

        Matches rover_commander.pick_up_block() exactly.
        """
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

            # Step 5: Lookup block transform (with retry)
            self.get_logger().info("🔍 Looking up block transform...")
            tf = self.lookup_transform_with_retry(self.base_frame, self.block_frame)

            # Log the raw block position
            raw_x = tf.transform.translation.x
            raw_y = tf.transform.translation.y
            raw_z = tf.transform.translation.z
            self.get_logger().info(
                f"📍 Block TF (raw): x={raw_x:.4f}, y={raw_y:.4f}, z={raw_z:.4f}"
            )

            # Get block orientation as quaternion
            q_block = [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ]

            block_euler = euler_from_quaternion(q_block)
            self.get_logger().info(
                f"🔄 Block quat: x={q_block[0]:.4f}, y={q_block[1]:.4f}, "
                f"z={q_block[2]:.4f}, w={q_block[3]:.4f}"
            )
            self.get_logger().info(
                f"🔄 Block euler (deg): roll={np.degrees(block_euler[0]):.1f}°, "
                f"pitch={np.degrees(block_euler[1]):.1f}°, "
                f"yaw={np.degrees(block_euler[2]):.1f}°"
            )

            # Step 6: Determine gripper orientation
            if self.USE_FIXED_ORIENTATION:
                self.get_logger().info("📐 Using FIXED orientation (ignoring block TF rotation)")
                q_fixed = quaternion_from_euler(
                    self.FIXED_ORIENTATION_ROLL,
                    self.FIXED_ORIENTATION_PITCH,
                    self.FIXED_ORIENTATION_YAW,
                )
            else:
                self.get_logger().info(
                    f"📐 Applying orientation correction: "
                    f"roll={self.ORIENTATION_CORRECTION_ROLL:.4f}, "
                    f"pitch={self.ORIENTATION_CORRECTION_PITCH:.4f}, "
                    f"yaw={self.ORIENTATION_CORRECTION_YAW:.4f}"
                )
                q_correction = quaternion_from_euler(
                    self.ORIENTATION_CORRECTION_ROLL,
                    self.ORIENTATION_CORRECTION_PITCH,
                    self.ORIENTATION_CORRECTION_YAW,
                )
                q_fixed = quaternion_multiply(q_block, q_correction)

            gripper_euler = euler_from_quaternion(q_fixed)
            self.get_logger().info(
                f"✅ Final gripper euler (deg): roll={np.degrees(gripper_euler[0]):.1f}°, "
                f"pitch={np.degrees(gripper_euler[1]):.1f}°, "
                f"yaw={np.degrees(gripper_euler[2]):.1f}°"
            )

            # Step 7: Compute target position with offsets
            tx = raw_x + self.BLOCK_X_OFFSET
            ty = raw_y + self.BLOCK_Y_OFFSET
            tz = raw_z
            self.get_logger().info(
                f"📍 Block TF (with offsets): x={tx:.4f}, y={ty:.4f}, z={tz:.4f}"
            )

            # Step 8: Approach (above block)
            self.get_logger().info(
                f"📍 Moving to approach position (z + {self.APPROACH_HEIGHT:.4f})..."
            )
            self.move_to_pose(tx, ty, tz + self.APPROACH_HEIGHT, q_fixed, self.APPROACH_DURATION)

            # Step 9: Lower to grasp position
            self.get_logger().info(
                f"📍 Moving down to grasp position (z + {self.GRASP_HEIGHT:.4f})..."
            )
            self.move_to_pose(tx, ty, tz + self.GRASP_HEIGHT, q_fixed, self.GRASP_DURATION)

            # Step 10: Close gripper
            self.get_logger().info("✊ Closing gripper...")
            self.control_gripper(False)

            # Step 11: Lift block
            self.get_logger().info(f"📍 Lifting block (z + {self.LIFT_HEIGHT:.4f})...")
            self.move_to_pose(tx, ty, tz + self.LIFT_HEIGHT, q_fixed, self.LIFT_DURATION)

            self.get_logger().info("✅ Block pickup complete!")
            return True

        except Exception as e:
            self.get_logger().error(f"❌ Pickup error: {e}")
            return False

    # ------------------------------------------------------------------
    # Place sequence (mirrors rover_commander.place_block_on_plate)
    # ------------------------------------------------------------------

    def place_block_on_plate(self):
        """Execute the sequence to place the held block onto the plate.

        Uses hardcoded placement coordinates (PLACE_X/Y/Z) derived from
        /actual_end_effector_pose captured while the arm was at the correct
        placement position.

        To re-tune:
            1. Move the arm to the desired placement position manually.
            2. Run: ros2 topic echo /actual_end_effector_pose --once
            3. Update PLACE_X, PLACE_Y, PLACE_Z at the top of this class
               AND in rover_commander.RoverCommander.
        """
        self.get_logger().info("🤖 Starting block placement sequence...")

        try:
            q_placement = quaternion_from_euler(
                self.PLACE_ROLL, self.PLACE_PITCH, self.PLACE_YAW
            )

            tx = self.PLACE_X
            ty = self.PLACE_Y
            tz = self.PLACE_Z

            self.get_logger().info(
                f"📍 Placement target (panda_link0): "
                f"x={tx:.4f}, y={ty:.4f}, z={tz:.4f}"
            )

            # 1. Hover above plate
            hover_z = tz + self.PLACE_HOVER_OFFSET
            self.get_logger().info(
                f"📍 Hovering over plate... IK target: ({tx:.4f}, {ty:.4f}, {hover_z:.4f})"
            )
            self.move_to_pose(tx, ty, hover_z, q_placement, 3.0)

            # 2. Lower to release height
            drop_z = tz + self.PLACE_DROP_OFFSET
            self.get_logger().info(
                f"📍 Lowering block... IK target: ({tx:.4f}, {ty:.4f}, {drop_z:.4f})"
            )
            self.move_to_pose(tx, ty, drop_z, q_placement, 2.0)

            # 3. Open gripper to release block
            self.get_logger().info("👐 Releasing block...")
            self.control_gripper(True)

            # 4. Retract arm upward
            retract_z = tz + self.PLACE_RETRACT_OFFSET
            self.get_logger().info(
                f"📍 Retracting arm... IK target: ({tx:.4f}, {ty:.4f}, {retract_z:.4f})"
            )
            self.move_to_pose(tx, ty, retract_z, q_placement, 2.0)

            self.get_logger().info("✅ Placement complete!")
            return True

        except Exception as e:
            self.get_logger().error(f"❌ Placement error: {e}")
            return False

    # ------------------------------------------------------------------
    # Test entry point
    # ------------------------------------------------------------------

    def run_test(self):
        """Run the pick test once; node stays alive for post-run inspection."""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("🚀 ARM PICK-PLACE TEST")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Steps:")
        self.get_logger().info("  1. Reset environment  (pub /reset_env)")
        self.get_logger().info("  2. Spawn block        (pub /spawn_blocc)")
        self.get_logger().info("  3. Pick up block      (IK + arm trajectory)")
        self.get_logger().info("  4. Place block        (hardcoded PLACE_X/Y/Z)")
        self.get_logger().info("  5. Stow arm           (neutral pose)")
        self.get_logger().info("=" * 60 + "\n")

        # Wait for IK service
        self.get_logger().info("⏳ Waiting for IK service...")
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ IK service not available! Is MoveIt running?")
            return
        self.get_logger().info("✅ IK service available")

        # Allow other nodes to be ready
        time.sleep(2.0)

        # --- Step A: Pick ---
        pick_ok = self.pick_up_block()
        if not pick_ok:
            self.get_logger().error("\n❌ PICK FAILED — aborting (check pick offsets)")
            return

        self.get_logger().info("\n✅ Pick succeeded — starting placement...")
        time.sleep(0.5)

        # --- Step B: Place ---
        place_ok = self.place_block_on_plate()
        if not place_ok:
            self.get_logger().error("\n❌ PLACE FAILED — check PLACE_X/Y/Z offsets")
        else:
            self.get_logger().info("\n✅ Place succeeded!")

        # --- Step C: Stow arm for travel ---
        self.get_logger().info("📍 Stowing arm...")
        self.move_to_pose(0.3, 0.0, 0.5, [1.0, 0.0, 0.0, 0.0], 2.0)

        if pick_ok and place_ok:
            self.get_logger().info("\n🎉 TEST PASSED — full pick-and-place complete!")
        else:
            self.get_logger().error("\n❌ TEST INCOMPLETE — one or more steps failed")

        self.get_logger().info("\nNode stays alive for inspection. Press Ctrl+C to exit.\n")
        self.get_logger().info(
            "Copy updated offsets to rover_commander.RoverCommander class constants."
        )


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ArmPickPlaceTest()

        # Run the test in a background thread so rclpy.spin() can process
        # IK service callbacks on the main thread.
        test_thread = threading.Thread(target=node.run_test, daemon=True)
        test_thread.start()

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
