#!/home/xavie/venv_thesis/bin/python3
"""
Standalone ROS2 node that uses trained behavior cloning model to control the robot arm.
Subscribes to joint states, predicts target poses, and publishes them to unity_target_pose.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import os



class BehaviorCloningModel(nn.Module):
    """Neural network model for behavior cloning (same as training script)"""
    def __init__(self, state_dim=7, action_dim=7, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class BCController(Node):
    def __init__(self, model_path="robot_bc_model.pth"):
        super().__init__('bc_controller')
        
        # Load trained model
        self.load_model(model_path)
        
        # ROS2 setup
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.pose_pub = self.create_publisher(
            Pose,
            '/bc_target_pose',
            10
        )
        
        # Control parameters
        self.current_joints = None
        self.last_prediction = None
        self.prediction_threshold = 0.0001  # Only publish if prediction changed significantly
        self.control_rate = 10.0  # Hz
        
        # Create timer for periodic control
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_callback
        )
        
        # Control flags
        self.autonomous_mode = True
        
        self.get_logger().info('BC Controller started. Press Ctrl+C to stop.')
        self.get_logger().info('Send "start" or "stop" commands to /bc_commands topic to control autonomous mode.')
        

        self.is_expert_active = False
        self.expert_active_sub = self.create_subscription(
            Bool,
            '/is_expert_active',
            self.is_expert_active_callback,
            10
        )
    
    def load_model(self, model_path):
        """Load the trained behavior cloning model and scalers"""
        model_file = Path(model_path)
        
        if not model_file.exists():
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load saved model data
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
        
        # Initialize model with saved config
        config = checkpoint['model_config']
        self.model = BehaviorCloningModel(
            state_dim=config['state_dim'],
            action_dim=config['action_dim']
        )   
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        
        # Load scalers
        self.state_scaler = checkpoint['state_scaler']
        self.action_scaler = checkpoint['action_scaler']
        
        self.get_logger().info(f'Model loaded successfully from {model_path}')
        self.get_logger().info(f'State dim: {config["state_dim"]}, Action dim: {config["action_dim"]}')
    
    def joint_callback(self, msg):
        """Callback for joint state updates"""
        # Extract first 7 joint positions (Panda arm joints)
        if len(msg.position) >= 7:
            self.current_joints = np.array(msg.position[:7], dtype=np.float32)
        else:
            self.get_logger().warn(f"Received joint state with {len(msg.position)} joints, expected 7+")

    def is_expert_active_callback(self, msg):
        self.is_expert_active = msg
    
    def predict_target_pose(self, joint_positions):
        """Use trained model to predict target end-effector pose"""
        # Normalize input using saved scaler
        joints_normalized = self.state_scaler.transform(joint_positions.reshape(1, -1))
        
        # Predict with model
        with torch.no_grad():
            joints_tensor = torch.FloatTensor(joints_normalized)
            prediction = self.model(joints_tensor).numpy()
        
        # Denormalize output
        target_pose = self.action_scaler.inverse_transform(prediction)[0]
        
        return target_pose  # [x, y, z, qx, qy, qz, qw]
    
    def control_callback(self):
        """Main control loop - runs at fixed rate"""
        if not self.autonomous_mode:
            return
            
        if self.current_joints is None:
            self.get_logger().warn('No joint states received yet')
            return
        
        try:
            # Get prediction from model
            predicted_pose = self.predict_target_pose(self.current_joints)
            
            # Check if prediction changed significantly
            if self.last_prediction is not None:
                pose_diff = np.linalg.norm(predicted_pose[:3] - self.last_prediction[:3])
                if pose_diff < self.prediction_threshold:
                    return  # Skip publishing if change is too small
            
            # Create and publish Pose message
            pose_msg = Pose()
            
            # Position
            pose_msg.position.x = float(predicted_pose[0])
            pose_msg.position.y = float(predicted_pose[1])
            pose_msg.position.z = float(predicted_pose[2])
            
            # Orientation (quaternion)
            pose_msg.orientation.x = float(predicted_pose[3])
            pose_msg.orientation.y = float(predicted_pose[4])
            pose_msg.orientation.z = float(predicted_pose[5])
            pose_msg.orientation.w = float(predicted_pose[6])
            
            # Publish
            self.pose_pub.publish(pose_msg)
            self.last_prediction = predicted_pose.copy()
            
            # Debug info
            self.get_logger().info(f'Published target: ({predicted_pose[0]:.3f}, {predicted_pose[1]:.3f}, {predicted_pose[2]:.3f})')
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')
    
    def start_autonomous(self):
        """Start autonomous control"""
        self.autonomous_mode = True
        self.get_logger().info('Autonomous control STARTED')
    
    def stop_autonomous(self):
        """Stop autonomous control"""
        self.autonomous_mode = False
        self.get_logger().info('Autonomous control STOPPED')

def main(args=None):
    rclpy.init(args=args)

    package_name = 'autonomy_bc'
    package_share_directory = get_package_share_directory(package_name)
    model_path = os.path.join(package_share_directory, 'models', 'robot_bc_model.pth')

    
    try:
        # Create controller node
        controller = BCController(model_path=model_path)
        
        print("\n=== BC Controller Ready ===")
        print("Commands:")
        print("  - The robot will start in STOPPED mode")
        print("  - Publish any message to /bc_commands to toggle start/stop")
        print("  - Or call controller.start_autonomous() programmatically")
        print("  - Press Ctrl+C to exit")
        print("\nTo start immediately, uncomment the line below:")
        print("# controller.start_autonomous()")
        
        controller.start_autonomous()
        
        # Spin the node
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        print("\nShutting down BC Controller...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()