#!/usr/bin/env python3
"""
Orchestrate Antenna Grid Deployment

This script coordinates the RoverCommander to deploy antennas at specified sites.
It can be extended to generate full 16x16 grids or other deployment patterns.

Usage:
    ros2 run driving_package orchestrate_deployment
"""

import rclpy
from rclpy.node import Node
from driving_package.rover_commander import RoverCommander
import threading
import time
from std_msgs.msg import Empty


class DeploymentOrchestrator(Node):
    def __init__(self):
        super().__init__('deployment_orchestrator')
        
        # Create the rover commander (but don't let it auto-start its sequence)
        self.commander = RoverCommander()
        
        # Override the sequence_started flag so it doesn't auto-run
        self.commander.sequence_started = True  # Prevent auto-start
        
        self.get_logger().info("🚀 Deployment Orchestrator initialized")
        self.get_logger().info("Waiting for plate locations from Unity before starting deployment...")
        
        # Store deployment sites
        self.deployment_sites = []
        self.deployment_started = False
        
    def set_deployment_sites(self, sites: list):
        """Set the list of sites for deployment."""
        self.deployment_sites = sites
        self.get_logger().info(f"📋 Loaded {len(sites)} deployment sites")
        
    def start_deployment(self):
        """Start the deployment sequence in a background thread."""
        if self.deployment_started:
            self.get_logger().warn("Deployment already started!")
            return
            
        self.deployment_started = True
        thread = threading.Thread(target=self._run_deployment, daemon=True)
        thread.start()
        
    def _run_deployment(self):
        """Run the deployment sequence."""
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("🚀 STARTING ANTENNA ROW DEPLOYMENT")
        self.get_logger().info("="*50 + "\n")
        
        if not self.deployment_sites:
            self.get_logger().error("No deployment sites loaded!")
            return
        
        total_sites = len(self.deployment_sites)
        successful_sites = 0
        
        # Each deployment site is a 3-point sequence
        for i, site in enumerate(self.deployment_sites):
            self.get_logger().info(f"\n{'#'*50}")
            self.get_logger().info(f"### DEPLOYMENT SITE {i+1}/{total_sites} ###")
            self.get_logger().info(f"{'#'*50}")

            
            # deploy_grid expects [rope_start, preamp, rope_end]
            antennas_deployed = self.commander.deploy_grid(site)
            
            if antennas_deployed > 0:
                successful_sites += 1
                self.get_logger().info(f"✅ Site {i+1} complete: {antennas_deployed} antenna(s) deployed")
            else:
                self.get_logger().error(f"❌ Site {i+1} failed")
        
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info(f"🎉 ROW DEPLOYMENT COMPLETE: {successful_sites}/{total_sites} sites successful")
        self.get_logger().info("="*50 + "\n")


def generate_grid(origin_x: float, origin_z: float, y_height: float, 
                  spacing: float, rows: int = 16, cols: int = 16) -> list:
    """
    Generate a grid of deployment sites.
    
    Args:
        origin_x: Starting X coordinate
        origin_z: Starting Z coordinate  
        y_height: Y coordinate (height) for all sites
        spacing: Distance between sites
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        List of [x, y, z] coordinates
    """
    sites = []
    for row in range(rows):
        for col in range(cols):
            x = origin_x + col * spacing
            z = origin_z + row * spacing
            sites.append([x, y_height, z])
    return sites


def main(args=None):
    rclpy.init(args=args)
    
    # Create orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # ==================== CONFIGURATION ====================
    # Each deployment site is a 3-point sequence: [rope_start, preamp_placement, rope_end]
    # The rover will:
    #   1. Go to rope_start and begin deploying rope
    #   2. Go to preamp_placement and do pick & place
    #   3. Go to rope_end and stop deploying rope
    
    # 4 deployment sites in a row (adjust coordinates as needed)
    row_of_sites = [
        # Site 1
        [
            [405.0, 18.0, 255.0],  # rope start
            [410.0, 18.0, 255.0],  # preamp placement
            [415.0, 18.0, 255.0],  # rope end
        ],
        # Site 2
        [
            [420.0, 18.0, 255.0],  # rope start
            [425.0, 18.0, 255.0],  # preamp placement
            [430.0, 18.0, 255.0],  # rope end
        ],
        # Site 3
        [
            [435.0, 18.0, 255.0],  # rope start
            [440.0, 18.0, 255.0],  # preamp placement
            [445.0, 18.0, 255.0],  # rope end
        ],
        # Site 4
        [
            [450.0, 18.0, 255.0],  # rope start
            [455.0, 18.0, 255.0],  # preamp placement
            [460.0, 18.0, 255.0],  # rope end
        ],
    ]
    
    orchestrator.set_deployment_sites(row_of_sites)
    
    # ==================== START DEPLOYMENT ====================
    orchestrator.get_logger().info("\n========================================")
    orchestrator.get_logger().info("=== ANTENNA DEPLOYMENT ORCHESTRATOR ===")
    orchestrator.get_logger().info("========================================")
    orchestrator.get_logger().info(f"Sites to deploy: {len(orchestrator.deployment_sites)}")
    orchestrator.get_logger().info("Starting deployment in 3 seconds...")
    
    # Create executor to spin both nodes
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(orchestrator)
    executor.add_node(orchestrator.commander)
    
    # Start deployment after a short delay (allow ROS connections to establish)
    def delayed_start():
        time.sleep(3.0)
        orchestrator.start_deployment()
    
    start_thread = threading.Thread(target=delayed_start, daemon=True)
    start_thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.destroy_node()
        orchestrator.commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
