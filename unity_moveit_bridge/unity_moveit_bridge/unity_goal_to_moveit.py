#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile

class UnityMoveItBridge(Node):
    def __init__(self):
        super().__init__('unity_moveit_bridge')

        # Callback group allows concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to Unity Pose goals
        qos = QoSProfile(depth=10)
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            qos,
            callback_group=self.callback_group
        )

        # Action client to MoveIt2 MoveGroup action server
        self._action_client = ActionClient(
            self,
            MoveGroup,
            'panda_arm/move_action',  # Make sure this matches your MoveIt2 launch
            callback_group=self.callback_group
        )

        self.get_logger().info("Unity → MoveIt2 bridge node started")

    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(
            f"Received Unity goal: x={msg.pose.position.x:.3f}, "
            f"y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}"
        )

        # Build MoveGroup goal
        move_goal = MoveGroup.Goal()
        move_goal.request.workspace_parameters.min_corner.x = -1.0
        move_goal.request.workspace_parameters.min_corner.y = -1.0
        move_goal.request.workspace_parameters.min_corner.z = 0.0
        move_goal.request.workspace_parameters.max_corner.x = 1.0
        move_goal.request.workspace_parameters.max_corner.y = 1.0
        move_goal.request.workspace_parameters.max_corner.z = 1.5
        move_goal.request.start_state.is_diff = True
        move_goal.request.goal_constraints.append(
            MoveGroup.Goal().request.goal_constraints[0]  # placeholder for now
        )

        # Normally, you'd convert PoseStamped → MoveIt GoalConstraint
        # For simplicity, we’ll use move_group_python_interface tutorials as reference

        # Wait for server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available")
            return

        # Send goal
        send_goal_future = self._action_client.send_goal_async(move_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by MoveIt2")
            return

        self.get_logger().info("Goal accepted by MoveIt2, executing...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info("MoveIt2 execution finished")

def main(args=None):
    rclpy.init(args=args)
    node = UnityMoveItBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
