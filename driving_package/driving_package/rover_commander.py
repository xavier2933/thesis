#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool, Empty
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import quaternion_multiply, quaternion_from_euler
import tf2_ros
import numpy as np
import time
import threading

class RoverCommander(Node):
    # ==================== TUNABLE BLOCK PICKUP OFFSETS ====================
    # Adjust these values when the object (block) changes
    
    # Block pickup offsets (relative to block TF)
    BLOCK_X_OFFSET = -0.04      # X offset from block center
    _dy = 398.1286 - 398.0723
    BLOCK_Y_OFFSET = -0.012 + _dy  # Y offset from block center
    
    _dz = 20.12106 - 19.97276 - 0.06
    
    # Pickup heights (relative to block Z)
    APPROACH_HEIGHT = 0.13 + _dz + 0.02    # Height for initial approach
    GRASP_HEIGHT = 0.07 + _dz + 0.02       # Height for grasping
    LIFT_HEIGHT = 0.15 + _dz        # Height after lifting
    
    # Gripper orientation correction (Euler angles in radians)
    ORIENTATION_CORRECTION_ROLL = 0.0                # Rotation around X
    ORIENTATION_CORRECTION_PITCH = -1.57079632679    # -90 degrees around Y
    ORIENTATION_CORRECTION_YAW = 0.0                 # Rotation around Z
    
    # ======================================================================
    
    def __init__(self):
        super().__init__('rover_commander')
        
        # --- Arm Control Config ---
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.block_frame = "block"
        self.plate_frame = "target_plate" # Broadcasted by Unity PlateTFBroadcaster
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # IK Client
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        
        # --- Publishers ---
        self.move_cmd_pub = self.create_publisher(Int32, 'rover/move_command', 10)
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        self.gripper_aut_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.rope_pub = self.create_publisher(Bool, 'rover/deploy_rope', 10)
        self.waypoint_pub = self.create_publisher(Pose, 'rover/waypoint', 10)
        self.spawn_pub = self.create_publisher(Empty, '/spawn_blocc', 10)

        
        # --- Subscribers ---
        self.plate_sub = self.create_subscription(PoseArray, 'rover/plate_locations', self.plate_locations_callback, 10)
        self.status_sub = self.create_subscription(String, 'rover/status', self.status_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        
        # --- State ---
        self.plate_positions = []
        self.current_plate_index = -1
        self.is_at_plate = False
        self.is_navigating = False
        self.rover_position = [0.0, 0.0, 0.0]
        self.latest_joints = None
        self.sequence_started = False 

        self.get_logger().info("[ROVER] Commander node initialized")
        
        self.get_logger().info("[ROVER] Waiting for plate locations from Unity...")

    def drive_distance_with_rope(self, distance_meters, deploy_rope=True):
        """Drives until a certain distance is covered while rope state is set."""
        start_pos = np.array(self.rover_position)
        
        # Toggle rope
        self.rope_pub.publish(Bool(data=deploy_rope))
        self.get_logger().info(f"[ROPE] Deployment: {deploy_rope}. Driving {distance_meters}m...")

        distance_covered = 0.0
        while distance_covered < distance_meters:
            current_pos = np.array(self.rover_position)
            distance_covered = np.linalg.norm(current_pos - start_pos)
            time.sleep(0.1) # Check at 10Hz
        
        self.get_logger().info(f"[ROPE] Target distance {distance_meters}m reached.")


    def send_waypoint(self, x, y, z):
        """Publishes a raw coordinate to Unity."""
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        # Identity orientation
        msg.orientation.w = 1.0
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"[WAYPOINT] Sent: {x}, {y}, {z}")

    def wait_for_waypoint_arrival(self, target_pos, tolerance=0.5):
        """Blocks until rover is near a specific coordinate."""
        self.get_logger().info(f"⏳ Navigating to {target_pos}...")
        while True:
            curr = np.array(self.rover_position)
            dist = np.linalg.norm(curr - np.array(target_pos))
            if dist < tolerance:
                self.get_logger().info("✅ Reached Waypoint.")
                return True
            time.sleep(0.1)

    # ==================== PUBLIC API FOR GRID DEPLOYMENT ====================
    # These methods are designed to be called from a higher-level orchestration script
    
    def go_to_site(self, x: float, y: float, z: float) -> bool:
        """
        Navigate rover to a specific site. Blocks until arrival.
        
        Args:
            x, y, z: World coordinates of the target site
            
        Returns:
            True if arrived successfully, False on timeout/failure
        """
        self.get_logger().info(f"📍 Navigating to site ({x:.1f}, {y:.1f}, {z:.1f})...")
        self.send_waypoint(x, y, z)
        return self.wait_for_unity_arrival()
    
    def deploy_antenna_at_current_site(self) -> bool:
        """
        Deploy an antenna at the rover's current location.
        Assumes rover has already arrived (Unity broadcasts target_plate TF).
        
        Returns:
            True if successful, False otherwise
        """
        self.get_logger().info("� Deploying antenna at current site...")
        return self.pick_and_place_at_waypoint()
    
    def deploy_antenna_at_site(self, x: float, y: float, z: float) -> bool:
        """
        Convenience method: Navigate to site AND deploy antenna.
        
        Args:
            x, y, z: World coordinates of the deployment site
            
        Returns:
            True if both navigation and deployment succeeded
        """
        self.get_logger().info(f"\n{'='*40}\n� DEPLOY ANTENNA AT ({x:.1f}, {y:.1f}, {z:.1f})\n{'='*40}")
        
        if not self.go_to_site(x, y, z):
            self.get_logger().error(f"❌ Failed to reach site ({x:.1f}, {y:.1f}, {z:.1f})")
            return False
        
        if not self.deploy_antenna_at_current_site():
            self.get_logger().error(f"❌ Failed to deploy at site ({x:.1f}, {y:.1f}, {z:.1f})")
            return False
        
        self.get_logger().info(f"✅ Antenna deployed at ({x:.1f}, {y:.1f}, {z:.1f})")
        return True
    
    def set_rope(self, deploying: bool):
        """
        Control rope deployment state.
        
        Args:
            deploying: True to start deploying rope, False to stop
        """
        self.rope_pub.publish(Bool(data=deploying))
        state = "STARTED" if deploying else "STOPPED"
        self.get_logger().info(f"🪢 Rope deployment {state}")
    
    def deploy_grid(self, sites: list) -> int:
        """
        Deploy antennas at multiple sites in sequence.
        
        Sequence logic:
        - First waypoint: Navigate, then START rope (no arm action)
        - Middle waypoints: Navigate, then pick & place
        - Last waypoint: Navigate, then STOP rope (no arm action)
        
        Args:
            sites: List of [x, y, z] coordinates (minimum 3 required)
            
        Returns:
            Number of successfully deployed antennas (middle waypoints only)
        """
        if len(sites) < 3:
            self.get_logger().error("Need at least 3 waypoints (start, middle, end)")
            return 0
        
        self.get_logger().info(f"STARTING GRID DEPLOYMENT ({len(sites)} sites)")
        self.get_logger().info(f"   First site: Start rope")
        self.get_logger().info(f"   Middle sites ({len(sites)-2}): Pick and Place")
        self.get_logger().info(f"   Last site: Stop rope")
        
        success_count = 0
        
        for i, site in enumerate(sites):
            is_first = (i == 0)
            is_last = (i == len(sites) - 1)
            is_middle = not is_first and not is_last
            
            # Navigate to site
            self.get_logger().info(f"\n{'='*40}")
            if is_first:
                self.get_logger().info(f"SITE {i+1}/{len(sites)}: START POSITION")
            elif is_last:
                self.get_logger().info(f"SITE {i+1}/{len(sites)}: END POSITION")
            else:
                self.get_logger().info(f"SITE {i+1}/{len(sites)}: DEPLOYMENT POINT")
            self.get_logger().info(f"{'='*40}")
            
            if not self.go_to_site(*site):
                self.get_logger().error(f"Failed to reach site {i+1}")
                if not is_first:
                    self.set_rope(False)
                return success_count
            
            # First waypoint: Start rope
            if is_first:
                self.get_logger().info("Starting rope deployment...")
                self.set_rope(True)
            
            # Middle waypoints: Pick and Place
            elif is_middle:
                self.get_logger().info("Executing pick and place...")
                if self.deploy_antenna_at_current_site():
                    success_count += 1
                else:
                    self.get_logger().error(f"Pick and place failed at site {i+1}")
            
            # Last waypoint: Stop rope
            elif is_last:
                self.get_logger().info("Stopping rope deployment...")
                self.set_rope(False)
        
        self.get_logger().info(f"\n{'='*40}")
        self.get_logger().info(f"GRID DEPLOYMENT COMPLETE: {success_count}/{len(sites)-2} antennas deployed")
        self.get_logger().info(f"{'='*40}")
        return success_count
    
    # ==================== INTERNAL TEST METHOD ====================
    
    def _test_deploy_sequence(self):
        """Internal test: Deploy at 3 hardcoded waypoints."""
        pts = [
            [405.0, 18.0, 255.0],
            [412.0, 18.0, 255.0],
            [415.0, 18.0, 255.0]
        ]
        self.deploy_grid(pts)
    
    def pick_and_place_at_waypoint(self):
        """
        Execute pick and place sequence at the current waypoint.
        Unity broadcasts the target_plate TF after arrival.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Give Unity time to broadcast the TF
            self.get_logger().info("⏳ Waiting for target TF from Unity...")
            time.sleep(1.0)
            
            # Refresh TF buffer to ensure we have latest transforms
            self.refresh_tf()
            
            # Step 1: Pick up the block
            self.pick_up_block()
            time.sleep(0.5)
            
            # Step 2: Place block at the target location
            self.place_block_on_plate()
            
            # Step 3: Stow arm for travel
            self.get_logger().info("📍 Stowing arm for travel...")
            self.move_to_pose(0.3, 0.0, 0.5, [1.0, 0.0, 0.0, 0.0], 2.0)
            
            self.get_logger().info("✅ Pick and place complete at this waypoint!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Pick and place error: {e}")
            return False

    def wait_for_unity_arrival(self, timeout=60.0):
        """Blocks until Unity reports navigation complete (isNavigating = False).
        
        Two-phase wait:
        1. Wait for Unity to acknowledge and START navigation (isNavigating = True)
        2. Wait for Unity to report arrival (isNavigating = False)
        """
        start_time = time.time()
        
        # Phase 1: Wait for Unity to START navigating (acknowledge the waypoint)
        self.get_logger().info("⏳ Waiting for Unity to acknowledge waypoint...")
        while not self.is_navigating:
            if time.time() - start_time > timeout:
                self.get_logger().error("⏰ Timeout waiting for navigation to start!")
                return False
            time.sleep(0.1)
        
        self.get_logger().info("🚗 Unity navigation started, waiting for arrival...")
        
        # Phase 2: Wait for Unity to FINISH navigating (arrival)
        while self.is_navigating:
            if time.time() - start_time > timeout:
                self.get_logger().error("⏰ Timeout waiting for arrival!")
                return False
            time.sleep(0.1)
        
        self.get_logger().info("📍 Unity reported arrival.")
        return True
        
        
    def drive_pick_and_place(self, plate_index):
        """Full autonomous loop: Drive, Pick, Place."""
        self.get_logger().info(f"\n{'='*50}\n🚀 STARTING PICK-AND-PLACE SEQUENCE\n{'='*50}")

        # 1. Drive to the plate
        self.send_move_command(plate_index)
        if not self.wait_for_arrival(plate_index):
            return

        # 2. Pick up the block
        self.pick_up_block()
        time.sleep(1.0)

        # 3. Place on the plate
        self.place_block_on_plate()

        self.get_logger().info(f"\n{'='*50}\n🎉 MISSION ACCOMPLISHED\n{'='*50}")

    def place_block_on_plate(self):
        """Execute the sequence to place the held block onto the plate."""
        self.get_logger().info("🤖 Starting block placement sequence...")
        
        try:
            # Lookup plate transform (Provided by your new Unity script)
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.plate_frame, rclpy.time.Time())
            
            # Use a standard top-down orientation for placement
            # You can also use the plate's rotation if needed
            q_placement = quaternion_from_euler(3.14159, 0.0, 1.5707) # Adjust based on desired orientation

            # Plate position
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z

            # Log the raw TF position
            self.get_logger().info(f"📍 TARGET_PLATE TF (raw): x={tx:.4f}, y={ty:.4f}, z={tz:.4f}")

            xoffset = 0.02
            tx += xoffset

            # Log the adjusted position
            self.get_logger().info(f"📍 TARGET_PLATE TF (with offset): x={tx:.4f}, y={ty:.4f}, z={tz:.4f}")

            # 1. Move to hover position over plate
            hover_z = tz + 0.20
            self.get_logger().info(f"📍 Hovering over plate... IK target: ({tx:.4f}, {ty:.4f}, {hover_z:.4f})")
            self.move_to_pose(tx, ty, hover_z, q_placement, 3.0)

            # 2. Lower to plate surface
            # Note: 0.05 is an offset to account for block height so it doesn't clip
            lower_z = tz + 0.08
            self.get_logger().info(f"📍 Lowering block... IK target: ({tx:.4f}, {ty:.4f}, {lower_z:.4f})")
            self.move_to_pose(tx, ty, lower_z, q_placement, 2.0)

            # 3. Open Gripper
            self.get_logger().info("👐 Releasing block...")
            self.control_gripper(True)

            # 4. Retract Arm
            self.get_logger().info("📍 Retracting arm...")
            self.move_to_pose(tx, ty, tz + 0.25, q_placement, 2.0)

            self.get_logger().info("✅ Placement complete!")

        except Exception as e:
            self.get_logger().error(f"❌ Placement error: {e}")
    
    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.latest_joints = np.array(msg.position[:7], dtype=np.float32)
        
    def plate_locations_callback(self, msg):
        """Receive plate locations from Unity (sent once at startup)"""
        self.plate_positions = []
        for pose in msg.poses:
            pos = [pose.position.x, pose.position.y, pose.position.z]
            self.plate_positions.append(pos)
        
        self.get_logger().info(f"[ROVER] ✓ Received {len(self.plate_positions)} plate locations:")
        for i, pos in enumerate(self.plate_positions):
            self.get_logger().info(f"  Plate {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Start the orchestration sequence once we have plate positions
        if not self.sequence_started and len(self.plate_positions) > 0:
            self.sequence_started = True
            # thread = threading.Thread(target=self.drive_to_plate_pick_and_place, args=(0,), daemon=True)
            thread = threading.Thread(target=self.deploy_antenna, daemon=True)
            thread.start()

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
    
    def wait_for_arrival(self, plate_index, timeout=60.0):
        """
        Block until rover arrives at the specified plate.
        
        Args:
            plate_index: The plate index we're waiting to arrive at
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if arrived, False if timed out
        """
        self.get_logger().info(f"[ROVER] ⏳ Waiting for arrival at plate {plate_index}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_at_plate and self.current_plate_index == plate_index:
                return True
            time.sleep(0.1)
        
        self.get_logger().error(f"[ROVER] ⏰ Timeout waiting for arrival at plate {plate_index}")
        return False
    
    def drive_to_plate_pick_and_place(self, plate_index):
        """
        Orchestrate the full sequence: 
        1. Drive to plate
        2. Pick up block
        3. Place block on the plate we just arrived at
        """
        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"🚀 Starting Full Orchestration")
        self.get_logger().info(f"   Step 1: Drive to plate {plate_index}")
        self.get_logger().info(f"   Step 2: Pick up the block")
        self.get_logger().info(f"   Step 3: Place block on plate")
        self.get_logger().info(f"{'='*50}\n")
        
        # --- Step 1: Drive to the plate ---
        self.send_move_command(plate_index)
        
        # Wait for rover to arrive
        if not self.wait_for_arrival(plate_index):
            self.get_logger().error("[ROVER] ❌ Sequence aborted: Failed to reach plate")
            return
        
        self.get_logger().info(f"[ROVER] ✅ Arrived. Starting block pickup...")
        time.sleep(1.0) 
        
        # --- Step 2: Pick up the block ---
        # Note: This method handles the simulation reset and the grasp
        self.pick_up_block()
        
        # --- Step 3: Place the block ---
        self.get_logger().info(f"[ROVER] 🤖 Pickup successful. Starting placement on plate {plate_index}...")
        self.place_block_on_plate()
        
        # --- Step 4: Stow the arm for travel ---
        self.get_logger().info(f"[ROVER] 📍 Stowing arm for travel...")
        # Move to a neutral upright position (adjust x, y, z to your home pose)
        self.move_to_pose(0.3, 0.0, 0.5, [1.0, 0.0, 0.0, 0.0], 2.0)

        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"🎉 MISSION COMPLETE FOR PLATE {plate_index}")
        self.get_logger().info(f"{'='*50}\n")
    
    def get_status(self):
        """Get current rover status"""
        return {
            'current_plate': self.current_plate_index,
            'at_plate': self.is_at_plate,
            'navigating': self.is_navigating,
            'position': self.rover_position
        }

    # --- ARM CONTROL METHODS (from orchestrate_data_collection_copy_2.py) ---
    
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
        
        self.target_pub.publish(pose.pose)
        joints = self.get_ik_solution(pose)
        if joints:
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            pt = JointTrajectoryPoint(positions=joints, time_from_start=rclpy.duration.Duration(seconds=duration).to_msg())
            traj.points = [pt]
            self.arm_pub.publish(traj)
            time.sleep(duration + 0.5)
        else:
            self.get_logger().error("IK Failed")

    def pick_up_block(self):
        """Execute the block pickup sequence (reset, approach, grasp, lift)."""
        self.get_logger().info("🤖 Starting block pickup sequence...")
        
        # Reset environment first
        self.reset_environment()
        self.spawn_pub.publish(Empty())
        time.sleep(1.0)
        
        try:
            # Refresh TF buffer to handle simulation time reset
            self.refresh_tf()
            
            # Open gripper
            self.control_gripper(True)

            # Lookup block transform
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.block_frame, rclpy.time.Time())
            q_block = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
            
            # Apply orientation correction
            q_correction = quaternion_from_euler(
                self.ORIENTATION_CORRECTION_ROLL,
                self.ORIENTATION_CORRECTION_PITCH,
                self.ORIENTATION_CORRECTION_YAW
            )
            q_fixed = quaternion_multiply(q_block, q_correction)
            
            # Get block position with offsets
            tx = tf.transform.translation.x + self.BLOCK_X_OFFSET
            ty = tf.transform.translation.y + self.BLOCK_Y_OFFSET
            tz = tf.transform.translation.z
            
            # Execute pickup sequence
            self.get_logger().info("📍 Moving to approach position...")
            self.move_to_pose(tx, ty, tz + self.APPROACH_HEIGHT, q_fixed, 3.0)
            
            self.get_logger().info("📍 Moving down to grasp position...")
            self.move_to_pose(tx, ty, tz + self.GRASP_HEIGHT, q_fixed, 3.0)
            
            self.get_logger().info("✊ Closing gripper...")
            self.control_gripper(False)
            
            self.get_logger().info("📍 Lifting block...")
            self.move_to_pose(tx, ty, tz + self.LIFT_HEIGHT, q_fixed, 2.0)
            
            self.get_logger().info("✅ Block pickup complete!")
            
        except Exception as e:
            self.get_logger().error(f"❌ Pickup error: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        commander = RoverCommander()
        
        commander.get_logger().info("\n========================================")
        commander.get_logger().info("=== ROVER COMMANDER - AUTO MODE ===")
        commander.get_logger().info("========================================")
        commander.get_logger().info("Will automatically drive to plate 0 and pick up block")
        commander.get_logger().info("once plate positions are received from Unity.")
        commander.get_logger().info("\nWaiting for plate positions...\n")
        
        rclpy.spin(commander)
        
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()