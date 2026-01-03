#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time
import threading

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from control_msgs.action import GripperCommand
from tf_transformations import quaternion_multiply, quaternion_from_euler

import tf2_ros
import tf_transformations


class MoveArmToBlockTF(Node):
    def __init__(self):
        super().__init__('move_arm_to_block_tf')

        self.group_name = "panda_arm"
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"
        self.target_frame = "block"

        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7"
        ]

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # IK
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("⏳ Waiting for IK service...")

        self.last_gripper_position = None
        self.position_tolerance = 1e-3

        # Gripper
        self.gripper_client = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')
        
        # Trajectory publisher
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/panda_arm_controller/joint_trajectory",
            10
        )
        
        # --- NEW: Publisher for Recorder & Bridge ---
        self.target_pub = self.create_publisher(Pose, "/target_pose_ros", 10)
        
        # We publish to BOTH to ensure:
        # 1. The Recorder hears it (listening on /gripper_command)
        # 2. The Bridge hears it (listening on /gripper_cmd_aut when in auth mode)
        self.gripper_state_pub = self.create_publisher(Bool, "/gripper_command", 10)
        self.gripper_aut_pub = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.auth_mode_pub = self.create_publisher(Bool, "/autonomous_mode", 10)

        self.current_gripper_open = False
        self.create_timer(0.1, self.publish_gripper_state)

        self.get_logger().info("✅ Move-to-block TF node started")

    def publish_gripper_state(self):
        msg = Bool(data=self.current_gripper_open)
        self.gripper_state_pub.publish(msg)
        self.gripper_aut_pub.publish(msg)

    def control_gripper(self, open_gripper: bool):
        self.current_gripper_open = open_gripper
        # 1. Enable autonomous mode so bridge listens to our command
        self.auth_mode_pub.publish(Bool(data=True))
        
        # 2. Determine state
        # position = 0.08 if open_gripper else 0.0
        
        # 3. Publish to BOTH topics:
        msg = Bool(data=open_gripper)
        self.gripper_state_pub.publish(msg) # For Recorder
        self.gripper_aut_pub.publish(msg)   # For Bridge actuation
        
        self.get_logger().info(f"🖐️ Gripper {'opening' if open_gripper else 'closing'} (via topic)...")

        # 4. Wait for execution (mimic dumb_alg behavior)
        time.sleep(1.0)

        self.last_gripper_position = 0.08 if open_gripper else 0.0
        self.get_logger().info("✅ Gripper command sent.")

    def gripper_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return
        self.get_logger().info("Gripper goal accepted")

    def get_ik_solution(self, pose: PoseStamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.timeout.sec = 2
        req.ik_request.avoid_collisions = True

        event = threading.Event()
        result_holder = {"res": None, "exc": None}

        def done_callback(future):
            try:
                result_holder["res"] = future.result()
            except Exception as e:
                result_holder["exc"] = e
            event.set()

        future = self.ik_client.call_async(req)
        future.add_done_callback(done_callback)
        
        # Wait for callback to fire
        if not event.wait(timeout=5.0):
            self.get_logger().error("TIMEOUT: IK service call timed out")
            return None

        if result_holder["exc"]:
            self.get_logger().error(f"IK service call failed: {result_holder['exc']}")
            return None
            
        res = result_holder["res"]
        if res is None:
            self.get_logger().error("IK service returned None")
            return None

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"❌ IK failed with error code: {res.error_code.val}")
            return None

        return res.solution.joint_state.position[:7]

    def publish_trajectory(self, joints, duration):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        pt = JointTrajectoryPoint()
        pt.positions = joints
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        traj.points = [pt]
        traj.header.stamp = self.get_clock().now().to_msg()

        self.arm_pub.publish(traj)
        time.sleep(duration + 0.5) # Wait for execution

    def move_sequence(self):
        # Allow some time for TF buffer to fill
        time.sleep(1.0)
        
        # 1. Open Gripper
        self.control_gripper(open_gripper=True)
        time.sleep(1.0)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Common orientation
        q_block = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ]
        # +90° about Z to fix Unity → ROS convention mismatch
        q_correction = quaternion_from_euler(-1.57079632679, 0.0, 0.0)
        q_fixed = quaternion_multiply(q_correction, q_block)

        # Define poses
        # Format: (name, z_offset, duration)
        steps = [
            ("Approach", 0.13, 4.0),
            ("Drop",     0.07, 3.0),
            # Grasp happens here logic-wise
            ("Return",   0.13, 3.0),
        ]

        # 2. Iterate steps
        for i, (name, z_offset, duration) in enumerate(steps):
            
            # If we are at the Return step (index 2), we should have grasped already
            if name == "Return":
                self.control_gripper(open_gripper=False)
                time.sleep(1.0) # Ensure grasp is settled

            pose = PoseStamped()
            pose.header.frame_id = self.base_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            
            pose.pose.position.x = tf.transform.translation.x + 0.04
            pose.pose.position.y = tf.transform.translation.y - 0.01
            pose.pose.position.z = tf.transform.translation.z + z_offset

            pose.pose.orientation.x = q_fixed[0]
            pose.pose.orientation.y = q_fixed[1]
            pose.pose.orientation.z = q_fixed[2]
            pose.pose.orientation.w = q_fixed[3]
            
            # Publish Target for Recorder
            self.target_pub.publish(pose.pose)
            
            self.get_logger().info(f"Computing IK for {name}...")
            joints = self.get_ik_solution(pose)
            
            if joints is None:
                self.get_logger().error(f"Could not solve IK for {name}, aborting sequence.")
                return

            self.get_logger().info(f"🚀 Moving to {name}...")
            
            # Keep publishing target continuously during move? 
            # ideally yes, but publishing once at start of segment signals intent well enough for simple segments.
            self.target_pub.publish(pose.pose)
            
            self.publish_trajectory(joints, duration)

        self.get_logger().info("✅ Sequence complete: Approach -> Drop -> Grasp -> Return")
        
        # Reset autonomous mode
        self.auth_mode_pub.publish(Bool(data=False))
        self.get_logger().info("🔓 Autonomous mode disabled.")


def main():
    rclpy.init()
    node = MoveArmToBlockTF()

    # Spin in a background thread so we can use blocking calls in the main thread
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    spinner_thread = threading.Thread(target=executor.spin, daemon=True)
    spinner_thread.start()

    try:
        node.move_sequence()
        # Keep the script alive briefly
        time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
