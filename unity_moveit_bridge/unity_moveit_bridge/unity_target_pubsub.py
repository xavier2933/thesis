#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from std_msgs.msg import Header, Bool, Float32
import copy
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
import math
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
from tf_transformations import quaternion_multiply, quaternion_from_euler


class UnityMoveItTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('unity_moveit_trajectory_bridge')

        self.group_name = "panda_arm"
        self.end_effector_link = "panda_hand"
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7"
        ]

        self.last_pose = None
        self.last_joint_positions = None
        self.pose_tolerance = 1e-3
        self.orientation_tolerance = 1e-2
        self.movement_tolerance = 1e-3
        self.last_gripper_position = None
        self.position_tolerance = 1e-3

        self.gripper_client = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')

        self.wristAngle = 0.0
        self.wrist_sub = self.create_subscription(
            Float32,
            '/wrist_angle',
            self.wrist_callback,
            10
        )

        self.bc_wrist_sub = self.create_subscription(
            Float32,
            '/bc_wrist_angle',
            self.bc_wrist_callback,
            10
        )

        # Subscribe to the arbitrated target pose
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
            self.gripper_callback_manual,
            10
        )

        self.gripper_aut_sub = self.create_subscription(
            Bool, 
            '/gripper_cmd_aut', 
            self.gripper_callback_autonomous, 
            10
        )

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

        self.autonomous_mode = False
        self.mode_sub = self.create_subscription(Bool, '/autonomous_mode', self.mode_callback, 10)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1):
            self.get_logger().info("⏳ Waiting for IK service...")

        self.last_pose = None
        self.get_logger().info("✅ Unity → MoveIt trajectory bridge started.")

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

    def wrist_callback(self, msg: Float32):
        # Only update wrist angle from manual control (Unity)
        if not self.autonomous_mode:
            self.wristAngle = msg.data

    def bc_wrist_callback(self, msg: Float32):
        if self.autonomous_mode:
            self.wristAngle = msg.data

    def pose_callback(self, pose_msg: Pose):
        """
        Process target pose from either Unity or Dreamer.
        - Unity poses need coordinate transformation
        - Dreamer poses are already in ROS frame
        """
        ros_x = pose_msg.position.z
        ros_y = -pose_msg.position.x
        ros_z = pose_msg.position.y

        base_down_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])
        q = quaternion_from_euler(0, 0, math.radians(self.wristAngle))
        final_q = quaternion_multiply(q, base_down_q)

        safe_pose = Pose()
        safe_pose.position.x = ros_x
        safe_pose.position.y = ros_y
        safe_pose.position.z = ros_z
        safe_pose.orientation.x = final_q[0]
        safe_pose.orientation.y = final_q[1]
        safe_pose.orientation.z = final_q[2]
        safe_pose.orientation.w = final_q[3]
        self.get_logger().debug("👤 Processing Unity pose (transformed to ROS frame)")

        # Skip IK if pose hasn't changed significantly
        if self.last_pose is not None:
            pos_diff = np.array([
                abs(safe_pose.position.x - self.last_pose.position.x),
                abs(safe_pose.position.y - self.last_pose.position.y),
                abs(safe_pose.position.z - self.last_pose.position.z)
            ])
            ori_diff = np.abs(np.array([
                safe_pose.orientation.x - self.last_pose.orientation.x,
                safe_pose.orientation.y - self.last_pose.orientation.y,
                safe_pose.orientation.z - self.last_pose.orientation.z,
                safe_pose.orientation.w - self.last_pose.orientation.w
            ]))
            if np.all(pos_diff < self.pose_tolerance) and np.all(ori_diff < self.orientation_tolerance):
                self.get_logger().debug("⏸️ Skipping IK request: pose unchanged.")
                return

        self.last_pose = copy.deepcopy(safe_pose)

        # Call IK
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = self.group_name
        ik_req.ik_request.pose_stamped.header.frame_id = "world"
        ik_req.ik_request.pose_stamped.pose = safe_pose
        ik_req.ik_request.timeout.sec = 1
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        future.add_done_callback(self.ik_response_callback)

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
            mode = "DREAMER" if self.autonomous_mode else "UNITY"
            self.get_logger().info(f"📤 [{mode}] Published arm trajectory")

        except Exception as e:
            self.get_logger().error(f"IK service call failed: {e}")

    def mode_callback(self, msg: Bool):
        self.autonomous_mode = msg.data
        mode = "AUTONOMOUS (Dreamer)" if msg.data else "MANUAL (Unity)"
        self.get_logger().info(f"🔄 Control mode switched to: {mode}")

    def gripper_callback_manual(self, msg: Bool):
        if self.autonomous_mode:
            self.get_logger().debug("🚫 Ignoring manual gripper command (autonomous mode active).")
            return
        self._execute_gripper_command(msg.data)

    def gripper_callback_autonomous(self, msg: Bool):
        if not self.autonomous_mode:
            self.get_logger().debug("🚫 Ignoring autonomous gripper command (manual mode active).")
            return
        self._execute_gripper_command(msg.data)

    def _execute_gripper_command(self, open_gripper: bool):
        position = 0.08 if open_gripper else 0.0
        max_effort = 2.0

        if self.last_gripper_position is not None:
            if abs(position - self.last_gripper_position) < self.position_tolerance:
                self.get_logger().debug("⏸️ Skipping redundant gripper command.")
                return

        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("⚠️ Gripper action server not available!")
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self.gripper_client.send_goal_async(goal_msg)
        self.last_gripper_position = position
        mode = "DREAMER" if self.autonomous_mode else "UNITY"
        self.get_logger().info(f"🖐️ [{mode}] Gripper {'opened' if open_gripper else 'closed'}")


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