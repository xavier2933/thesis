#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        # Ensure this matches your Unity topic exactly
        self.topic_name = "/camera/rgb/image_raw/compressed"
        
        # Create subscription
        # The '10' here is the queue size (QoS)
        self.subscription = self.create_subscription(
            CompressedImage,
            self.topic_name,
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f'Subscribed to: {self.topic_name}')

    def listener_callback(self, msg):
        try:
            # 1. Convert ROS CompressedImage message to numpy array
            np_arr = np.frombuffer(msg.data, np.uint8)
            
            # 2. Decode to OpenCV image
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 3. Display
            cv2.imshow("WSL Robot View", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")

def main(args=None):
    rclpy.init(args=args)
    viewer = CameraViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        viewer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()