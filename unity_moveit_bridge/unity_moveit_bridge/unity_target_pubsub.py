#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from std_msgs.msg import Header
import copy
import time

class UnityMoveItTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('unity_moveit_trajectory_bridge')

        # Parameters
        self.group_name = "panda_arm"
        self.end_effector_link = "panda_hand"
        self.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3",
                            "panda_joint4", "panda_joint5", "panda_joint6",
                            "panda_joint7"]

        # Subscriber to Unity target poses
        self.subscription = self.create_subscription(
            Pose,
            # '/unity_target_pose',
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

        # IK service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for IK service...")

        # Publisher for joint trajectory
        self.trajectory_pub = self.create_publisher(JointTrajectory, 'panda_arm_controller/joint_trajectory', 10)

        # Trajectory queue
        self.trajectory_queue = []
        self.last_pose = None
        self.get_logger().info("✅ Unity → MoveIt trajectory bridge started.")

    def reset_callback(self, msg):
        """Handle reset commands from Unity"""
        # Convert to joint trajectory and publish
        traj = JointTrajectory()
        traj.joint_names = list(msg.name)
        
        pt = JointTrajectoryPoint()
        pt.positions = list(msg.position)
        pt.time_from_start.sec = 3  # 3 seconds to reach position
        pt.time_from_start.nanosec = 0
        
        traj.points = [pt]
        traj.header.stamp = self.get_clock().now().to_msg()
        
        self.trajectory_pub.publish(traj)
        self.get_logger().info("Arm reset command executed")

    def pose_callback(self, pose_msg: Pose):
        # Only add if moved significantly
        if self.last_pose:
            dx = pose_msg.position.x - self.last_pose.position.x
            dy = pose_msg.position.y - self.last_pose.position.y
            dz = pose_msg.position.z - self.last_pose.position.z
            if dx*dx + dy*dy + dz*dz < 0.00144:  # ~2 cm threshold
                return

        self.last_pose = copy.deepcopy(pose_msg)

        # ---- Unity → ROS coordinate transform ----
        # Unity: (X = right, Y = up, Z = forward)
        # ROS:   (X = forward, Y = left, Z = up)
        ros_x = pose_msg.position.z
        ros_y = -pose_msg.position.x
        ros_z = pose_msg.position.y

        # Call IK service
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = self.group_name
        ik_req.ik_request.pose_stamped.header.frame_id = "world"

        safe_pose = Pose()
        safe_pose.position.x = ros_x
        safe_pose.position.y = ros_y
        safe_pose.position.z = ros_z

        # For now, keep a neutral orientation
        safe_pose.orientation.w = 1.0
        safe_pose.orientation.x = 0.0
        safe_pose.orientation.y = 0.0
        safe_pose.orientation.z = 0.0

        ik_req.ik_request.pose_stamped.pose = safe_pose
        ik_req.ik_request.timeout.sec = 1
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        future.add_done_callback(self.ik_response_callback)


    def ik_response_callback(self, future):
        try:
            response = future.result()
            if response.error_code.val != 1:
                self.get_logger().warn("IK failed for target pose")
                return

            # Extract first 7 Panda joints (ignore finger joints)
            joint_positions = list(response.solution.joint_state.position)[:len(self.joint_names)]

            traj = JointTrajectory()
            traj.joint_names = self.joint_names

            pt = JointTrajectoryPoint()
            pt.positions = joint_positions
            pt.time_from_start.sec = 2   # give controller 2 seconds to reach target
            pt.time_from_start.nanosec = 0

            traj.points = [pt]

            # Optional: header timestamp (some controllers expect this)
            from std_msgs.msg import Header
            traj.header = Header()
            traj.header.stamp = self.get_clock().now().to_msg()

            self.trajectory_pub.publish(traj)
            self.get_logger().info(f"Published trajectory: {joint_positions}")

        except Exception as e:
            self.get_logger().error(f"IK service call failed: {e}")

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
