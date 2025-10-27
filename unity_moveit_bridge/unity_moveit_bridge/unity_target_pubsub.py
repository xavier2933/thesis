#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from std_msgs.msg import Header, Bool
import copy
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
import math


class UnityMoveItTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('unity_moveit_trajectory_bridge')

        # --- MoveIt Parameters ---
        self.group_name = "panda_arm"
        self.end_effector_link = "panda_hand"
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7"
        ]

        self.gripper_client = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')

        # --- Subscriptions ---
        self.pose_sub = self.create_subscription(
            Pose,
            '/target_pose',
            self.pose_callback,
            10
        )

        self.reset_sub = self.create_subscription(
            JointState,
            '/reset_arm_command',
            self.reset_callback,
            10
        )

        self.gripper_sub = self.create_subscription(
            Bool,
            '/gripper_command',
            self.gripper_callback,
            10
        )

        # --- Publishers ---
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            'panda_arm_controller/joint_trajectory',
            10
        )

        self.hand_pub = self.create_publisher(
            JointTrajectory,
            'panda_hand_controller/joint_trajectory',
            10
        )

        # --- IK Service ---
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1):
            self.get_logger().info("⏳ Waiting for IK service...")

        self.last_pose = None
        self.get_logger().info("✅ Unity → MoveIt trajectory bridge started.")

    # ---------------------------------------------------
    # 🦾 Reset the arm to a known joint configuration
    # ---------------------------------------------------
    def reset_callback(self, msg):
        traj = JointTrajectory()
        traj.joint_names = list(msg.name)

        pt = JointTrajectoryPoint()
        pt.positions = list(msg.position)
        pt.time_from_start.sec = 3
        traj.points = [pt]

        traj.header.stamp = self.get_clock().now().to_msg()
        self.arm_pub.publish(traj)
        self.get_logger().info("♻️ Arm reset command executed")

    def quat_to_euler(self, x, y, z, w):
        """Convert quaternion to euler angles (roll, pitch, yaw) in degrees"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


    def pose_callback(self, pose_msg: Pose):
        
        if self.last_pose:
            dx = pose_msg.position.x - self.last_pose.position.x
            dy = pose_msg.position.y - self.last_pose.position.y
            dz = pose_msg.position.z - self.last_pose.position.z
            if dx * dx + dy * dy + dz * dz < 0.0001:  # 2 cm threshold
                return
        self.last_pose = copy.deepcopy(pose_msg)

        # --- Unity → ROS coordinate conversion ---
        ros_x = pose_msg.position.z
        ros_y = -pose_msg.position.x
        ros_z = pose_msg.position.y

        # --- Convert orientation from Unity to ROS ---
        # Unity uses left-handed coordinates, ROS uses right-handed
        # This conversion depends on your Unity's coordinate frame setup
        ros_quat_x = 0.7071068
        ros_quat_y = -0.7071068
        ros_quat_z = 0.0
        ros_quat_w = 0.0

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = self.group_name
        ik_req.ik_request.pose_stamped.header.frame_id = "world"

        safe_pose = Pose()
        safe_pose.position.x = ros_x
        safe_pose.position.y = ros_y
        safe_pose.position.z = ros_z
        
        # Use the actual orientation from Unity instead of neutral
        safe_pose.orientation.x = ros_quat_x
        safe_pose.orientation.y = ros_quat_y
        safe_pose.orientation.z = ros_quat_z
        safe_pose.orientation.w = ros_quat_w
        
        ik_req.ik_request.pose_stamped.pose = safe_pose
        ik_req.ik_request.timeout.sec = 1
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        future.add_done_callback(self.ik_response_callback)

    # ---------------------------------------------------
    # 🧠 Handle IK response (for arm trajectory)
    # ---------------------------------------------------
    def ik_response_callback(self, future):
        try:
            response = future.result()
            if response.error_code.val != 1:
                self.get_logger().warn("⚠️ IK failed for target pose")
                return

            joint_positions = list(response.solution.joint_state.position)[:len(self.joint_names)]
            traj = JointTrajectory()
            traj.joint_names = self.joint_names

            pt = JointTrajectoryPoint()
            pt.positions = joint_positions
            pt.time_from_start.sec = 2
            traj.points = [pt]

            traj.header.stamp = self.get_clock().now().to_msg()
            self.arm_pub.publish(traj)
            self.get_logger().info(f"📤 Published arm trajectory: {joint_positions}")

        except Exception as e:
            self.get_logger().error(f"IK service call failed: {e}")

    # ---------------------------------------------------
    # ✋ Handle gripper open/close commands
    # ---------------------------------------------------
    def gripper_callback(self, msg: Bool):
        open_gripper = msg.data
        position = 0.04 if open_gripper else 0.0  # open ≈ 4cm, closed ≈ 0cm
        max_effort = 2.0

        # Build the goal
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        # Wait until the action server is ready
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper action server not available!")
            return

        # Send goal asynchronously
        self.gripper_client.send_goal_async(goal_msg)
        self.get_logger().info(f"🖐️ Gripper {'opened' if open_gripper else 'closed'}")



def main(args=None):
    rclpy.init(args=args)
    node = UnityMoveItTrajectoryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()