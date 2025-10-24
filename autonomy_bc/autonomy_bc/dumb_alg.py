import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import time

class DumbAlg(Node):
    def __init__(self):
        super().__init__("DumbAlg")

        self.bc_pub = self.create_publisher(Pose, '/bc_target_pose', 10)
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Run sequence after a short delay to ensure publishers are ready
        self.timer = self.create_timer(1.0, self.run_sequence)
        self.has_run = False

    def run_sequence(self):
        if self.has_run:
            return
        self.has_run = True

        # --- Move forward for 1 second ---
        self.get_logger().info("Moving forward...")
        twist = Twist()
        twist.linear.x = 0.5
        twist.angular.z = 0.0
        self.publish_for_duration(twist, 5.0)

        # --- Turn left for 1 second ---
        self.get_logger().info("Turning left...")
        twist.linear.x = 0.0
        twist.angular.z = 0.5
        self.publish_for_duration(twist, 2.0)

        # --- Stop ---
        self.get_logger().info("Stopping.")
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.move_pub.publish(twist)

    def publish_for_duration(self, twist_msg, duration):
        """Publish the same message repeatedly for a duration."""
        start_time = time.time()
        pub_rate = 10.0  # Hz
        period = 1.0 / pub_rate

        while (time.time() - start_time) < duration and rclpy.ok():
            self.move_pub.publish(twist_msg)
            time.sleep(period)  # throttle to 10 Hz


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
