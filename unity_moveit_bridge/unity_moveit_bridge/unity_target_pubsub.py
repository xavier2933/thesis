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
    np.float = float  # temporary patch for old packages
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

        self.gripper_client = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')

        self.wristAngle = 0.0
        self.wrist_sub = self.create_subscription(
            Float32,
            '/wrist_angle',
            self.wrist_callback,
            10
        )

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

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1):
            self.get_logger().info("⏳ Waiting for IK service...")

        self.last_pose = None
        self.get_logger().info("✅ Unity → MoveIt trajectory bridge started.")


    # Reset to position for IL training
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
        self.wristAngle = msg.data
        self.get_logger().info(f"Received wrist angle = {msg.data}")


    def pose_callback(self, pose_msg: Pose):

        self.last_pose = copy.deepcopy(pose_msg)

        # --- Unity → ROS coordinate conversion ---
        ros_x = pose_msg.position.z
        ros_y = -pose_msg.position.x
        ros_z = pose_msg.position.y

        base_down_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])
        q = quaternion_from_euler(0,0, math.radians(self.wristAngle))
        final_q = quaternion_multiply(q, base_down_q)

        ros_quat_x = pose_msg.orientation.z
        ros_quat_y = -pose_msg.orientation.x
        ros_quat_z = pose_msg.orientation.y
        ros_quat_w = pose_msg.orientation.w


        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = self.group_name
        ik_req.ik_request.pose_stamped.header.frame_id = "world"

        safe_pose = Pose()
        safe_pose.position.x = ros_x
        safe_pose.position.y = ros_y
        safe_pose.position.z = ros_z
        
        # Use the actual orientation from Unity instead of neutral
        safe_pose.orientation.x = final_q[0]
        safe_pose.orientation.y = final_q[1]
        safe_pose.orientation.z = final_q[2]
        safe_pose.orientation.w = final_q[3]
        
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
            self.get_logger().info(f"📤 Published arm trajectory: {joint_positions}")

        except Exception as e:
            self.get_logger().error(f"IK service call failed: {e}")


    def gripper_callback(self, msg: Bool):
        open_gripper = msg.data
        position = 0.08 if open_gripper else 0.0  # 0 < open < 0.9???
        max_effort = 2.0

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper action server not available!")
            return

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