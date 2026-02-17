#!/usr/bin/env python3
"""
LLM Task Planner for Rover Deployment (ReAct Style)

This script uses OpenAI function calling to let an LLM plan and execute
antenna deployments step-by-step.

Usage:
    ros2 run driving_package llm_orchestrator
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose
from driving_package.rover_commander import RoverCommander
import json
import re
import threading
import time
import os
from datetime import datetime
from openai import OpenAI

# Directory to save conversation logs
LOG_DIR = os.path.expanduser("~/thesis_ws/llm_logs")

# ============================================================================
# MISSION CONFIGURATION
# ============================================================================

DEPLOYMENT_SITES = [
    {
        "site_id": 1,
        "description": "Site 1 - First antenna in the row",
        "waypoints": {
            "rope_start": [405.0, 18.0, 255.0],
            "preamp": [410.0, 18.0, 255.0],
            "rope_end": [415.0, 18.0, 255.0],
        }
    },
    {
        "site_id": 2,
        "description": "Site 2 - Second antenna in the row",
        "waypoints": {
            "rope_start": [420.0, 18.0, 255.0],
            "preamp": [425.0, 18.0, 255.0],
            "rope_end": [430.0, 18.0, 255.0],
        }
    },
    {
        "site_id": 3,
        "description": "Site 3 - Third antenna in the row",
        "waypoints": {
            "rope_start": [435.0, 18.0, 255.0],
            "preamp": [440.0, 18.0, 255.0],
            "rope_end": [445.0, 18.0, 255.0],
        }
    },
    {
        "site_id": 4,
        "description": "Site 4 - Fourth antenna in the row",
        "waypoints": {
            "rope_start": [450.0, 18.0, 255.0],
            "preamp": [455.0, 18.0, 255.0],
            "rope_end": [460.0, 18.0, 255.0],
        }
    },
]

# ============================================================================
# TOOL DEFINITIONS (OpenAI Function Calling Schema)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deploy_antenna",
            "description": "Deploy an antenna at the specified site. The rover will navigate to the site's waypoints (rope_start -> preamp -> rope_end), deploying rope and placing the preamp antenna.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_id": {
                        "type": "integer",
                        "description": "The site ID to deploy at (1-4)"
                    }
                },
                "required": ["site_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_mission_status",
            "description": "Get the current mission status including which sites have been deployed and the rover's approximate location.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mission_complete",
            "description": "Call this when all antenna deployments are complete to end the mission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what was accomplished"
                    }
                },
                "required": ["summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_around_obstacle",
            "description": "Navigate the rover around a detected obstacle using curved paths. Only use this if the obstacle is AHEAD of the rover (obstacle X > rover X). Choose left or right based on which side has more clearance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obstacle_id": {
                        "type": "integer",
                        "description": "The ID of the obstacle to navigate around"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which side to swerve around the obstacle. In Unity coordinates: left = -Z, right = +Z."
                    }
                },
                "required": ["obstacle_id", "direction"]
            }
        }
    },
]

SYSTEM_PROMPT = """You are an autonomous rover mission planner for lunar antenna deployment.

MISSION OBJECTIVE:
Deploy antennas at 4 sites in sequence. Each site requires:
1. Navigate to rope_start and begin deploying rope
2. Navigate to preamp location and place the antenna
3. Navigate to rope_end and stop deploying rope

ROVER POSITION:
{rover_position}

AVAILABLE SITES:
{sites_info}

CURRENT OBSTACLES:
{obstacles_info}

RULES:
- Deploy sites in order (1 -> 2 -> 3 -> 4) unless blocked by obstacles
- After each deployment, check the mission status
- The rover travels along the +X axis. Only go around obstacles that are AHEAD (obstacle X > rover X)
- If an obstacle is behind you (obstacle X < rover X), ignore it
- When avoiding obstacles, choose left (-Z) or right (+Z) based on clearance
- Call mission_complete when all sites are deployed

Think step by step. After each action, evaluate the result and decide the next step."""


class LLMOrchestrator(Node):
    """ReAct-style LLM planner for rover deployment."""
    
    def __init__(self):
        super().__init__('llm_orchestrator')
        
        # ROS parameters
        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter('debug_mode').value
        
        # Initialize OpenAI client
        self.client = OpenAI()
        self.model = "gpt-5-nano"  # Can change to gpt-4o-mini for faster/cheaper
        
        # Initialize RoverCommander (pass debug_mode to skip arm operations)
        self.commander = RoverCommander(debug_mode=self.debug_mode)
        self.commander.sequence_started = True  # Prevent auto-start
        
        # Mission state
        self.deployed_sites = []
        self.deployment_history = []  # [{site_id, success, reason}, ...]
        self.current_obstacles = []  # Populated by /rock_detection subscriber
        self._next_obstacle_id = 1
        self.mission_active = False
        self.conversation_history = []
        
        # Deployment result from Unity
        self.pending_deployment_result = None
        self.deployment_site_pub = self.create_publisher(Int32, '/deployment_site_id', 10)
        self.deployment_result_sub = self.create_subscription(
            String,
            '/deployment_result',
            self.deployment_result_callback,
            10
        )
        
        # Rock detection subscriber
        self.rock_detection_sub = self.create_subscription(
            String,
            '/rock_detection',
            self.rock_detection_callback,
            10
        )
        
        # Curved goal publisher for obstacle avoidance
        self.curved_goal_pub = self.create_publisher(Pose, '/rover/curved_goal', 10)
        
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file = None
        
        self.get_logger().info("🤖 LLM Orchestrator initialized")
        self.get_logger().info(f"📁 Logs will be saved to: {LOG_DIR}")
        if self.debug_mode:
            self.get_logger().info("⚡ DEBUG MODE: arm operations will be skipped in RoverCommander")
    
    def deployment_result_callback(self, msg: String):
        """Receive deployment validation result from Unity."""
        # Format: "site_id,success,reason"
        try:
            parts = msg.data.split(',', 2)
            site_id = int(parts[0])
            success = parts[1].lower() == 'true'
            reason = parts[2] if len(parts) > 2 else ""
            
            self.pending_deployment_result = {
                "site_id": site_id,
                "success": success,
                "reason": reason
            }
            self.get_logger().info(f"📥 Received deployment result: site={site_id}, success={success}, reason={reason}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse deployment result: {e}")
    
    def rock_detection_callback(self, msg: String):
        """Receive rock detection from Unity and add to obstacles list.
        
        Expected format: 'Rock at (x, y, z) detected!'
        """
        try:
            match = re.search(r'\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', msg.data)
            if not match:
                self.get_logger().warn(f"Could not parse rock position from: {msg.data}")
                return
            
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            
            # Check for duplicates (within 2m of an existing obstacle)
            for o in self.current_obstacles:
                dist = ((o['x'] - x)**2 + (o['y'] - y)**2 + (o['z'] - z)**2) ** 0.5
                if dist < 2.0:
                    return  # Already tracked
            
            obstacle = {
                "id": self._next_obstacle_id,
                "x": x,
                "y": y,
                "z": z,
                "radius": 3.0,
                "description": f"Rock detected at ({x}, {y}, {z})"
            }
            self._next_obstacle_id += 1
            self.current_obstacles.append(obstacle)
            
            self.get_logger().info(f"🪨 Rock detected! Added obstacle id={obstacle['id']} at ({x}, {y}, {z})")
        except Exception as e:
            self.get_logger().error(f"Failed to parse rock detection: {e}")
        
    def get_sites_info(self) -> str:
        """Format site information for the system prompt."""
        lines = []
        for site in DEPLOYMENT_SITES:
            status = "✅ DEPLOYED" if site["site_id"] in self.deployed_sites else "⏳ PENDING"
            wp = site["waypoints"]
            lines.append(
                f"- Site {site['site_id']}: {site['description']} [{status}]\n"
                f"  Waypoints: rope_start={wp['rope_start']}, preamp={wp['preamp']}, rope_end={wp['rope_end']}"
            )
        return "\n".join(lines)
    
    def get_obstacles_info(self) -> str:
        """Format obstacle information for the system prompt."""
        if not self.current_obstacles:
            return "None detected"
        return "\n".join([
            f"- Obstacle id={o['id']}: {o['description']} (radius {o['radius']}m)"
            for o in self.current_obstacles
        ])
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with current state."""
        pos = self.commander.rover_position
        return SYSTEM_PROMPT.format(
            rover_position=f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) — traveling along +X axis",
            sites_info=self.get_sites_info(),
            obstacles_info=self.get_obstacles_info()
        )
    
    # ========================================================================
    # TOOL IMPLEMENTATIONS
    # ========================================================================
    
    def tool_deploy_antenna(self, site_id: int) -> str:
        """Execute antenna deployment at a site."""
        self.get_logger().info(f"🎯 LLM requested: deploy_antenna(site_id={site_id})")
        
        # Validate site_id
        if site_id < 1 or site_id > len(DEPLOYMENT_SITES):
            return json.dumps({
                "success": False,
                "error": f"Invalid site_id {site_id}. Must be 1-{len(DEPLOYMENT_SITES)}"
            })
        
        if site_id in self.deployed_sites:
            return json.dumps({
                "success": False,
                "error": f"Site {site_id} already deployed"
            })
        
        # Get site waypoints
        site = DEPLOYMENT_SITES[site_id - 1]
        wp = site["waypoints"]
        waypoint_list = [wp["rope_start"], wp["preamp"], wp["rope_end"]]
        
        self.get_logger().info(f"📍 Deploying at site {site_id}: {waypoint_list}")
        
        # Publish site_id to Unity BEFORE deployment so it knows which antenna to validate
        site_msg = Int32()
        site_msg.data = site_id
        self.deployment_site_pub.publish(site_msg)
        self.get_logger().info(f"📤 Published deployment site_id={site_id} to Unity")
        
        # Clear any old pending result
        self.pending_deployment_result = None
        
        # Execute deployment via RoverCommander
        try:
            antennas_deployed = self.commander.deploy_grid(waypoint_list, site_id)
    
            # Wait for Unity validation result
            if self.debug_mode:
                validation_result = {"success": True, "reason": "debug_mode: validation skipped"}
            else:
                validation_result = self.wait_for_deployment_result(site_id, timeout=45.0)
            
            # Record in deployment history (always, even on failure)
            self.deployed_sites.append(site_id)
            self.deployment_history.append({
                "site_id": site_id,
                "rover_success": antennas_deployed > 0,
                "validation": validation_result
            })
            

            # Build response
            obstacle_warning = None
            if self.current_obstacles:
                obstacle_warning = (f"{len(self.current_obstacles)} obstacle(s) detected on the field. "
                                    f"Check mission status for details and use go_around_obstacle if one is blocking your path.")
            
            if validation_result and validation_result.get("success"):
                return json.dumps({
                    "success": True,
                    "message": f"Site {site_id} deployed and validated successfully.",
                    "validation": validation_result.get("reason", ""),
                    "deployed_sites": self.deployed_sites,
                    "warning": obstacle_warning
                })
            elif validation_result:
                return json.dumps({
                    "success": False,
                    "message": f"Site {site_id} deployment FAILED validation.",
                    "reason": validation_result.get("reason", "Unknown"),
                    "deployed_sites": self.deployed_sites,
                    "note": "Continue to next site despite failure."
                })
            else:
                return json.dumps({
                    "success": True,
                    "message": f"Site {site_id} physically deployed ({antennas_deployed} antenna(s)), but no validation received from Unity.",
                    "deployed_sites": self.deployed_sites,
                    "warning": obstacle_warning
                })
                
        except Exception as e:
            self.deployment_history.append({
                "site_id": site_id,
                "rover_success": False,
                "validation": {"success": False, "reason": str(e)}
            })
            return json.dumps({
                "success": False,
                "error": f"Deployment error: {str(e)}"
            })
    
    def wait_for_deployment_result(self, site_id: int, timeout: float = 10.0) -> dict:
        """Wait for Unity to send deployment validation result."""
        self.get_logger().info(f"⏳ Waiting for Unity validation (timeout={timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.pending_deployment_result and self.pending_deployment_result.get("site_id") == site_id:
                result = self.pending_deployment_result
                self.pending_deployment_result = None
                return result
            time.sleep(0.1)
        
        self.get_logger().warn(f"⚠️ Timeout waiting for validation result for site {site_id}")
        return None

    
    def tool_get_mission_status(self) -> str:
        """Return current mission status."""
        self.get_logger().info("📊 LLM requested: get_mission_status()")
        
        total_sites = len(DEPLOYMENT_SITES)
        pending = [s["site_id"] for s in DEPLOYMENT_SITES if s["site_id"] not in self.deployed_sites]
        
        return json.dumps({
            "deployed_sites": self.deployed_sites,
            "pending_sites": pending,
            "progress": f"{len(self.deployed_sites)}/{total_sites}",
            "obstacles": self.current_obstacles,
            "mission_complete": len(pending) == 0
        })
    
    def tool_mission_complete(self, summary: str) -> str:
        """Mark mission as complete."""
        self.get_logger().info(f"🏁 LLM requested: mission_complete(summary='{summary}')")
        self.mission_active = False
        return json.dumps({
            "success": True,
            "message": "Mission marked as complete",
            "summary": summary,
            "final_deployed_sites": self.deployed_sites
        })
    
    def tool_go_around_obstacle(self, obstacle_id: int, direction: str = "left") -> str:
        """Navigate around an obstacle using two Bezier curve goals."""
        self.get_logger().info(f"🪨 LLM requested: go_around_obstacle(obstacle_id={obstacle_id}, direction={direction})")
        
        # Find the obstacle
        obstacle = None
        for o in self.current_obstacles:
            if o["id"] == obstacle_id:
                obstacle = o
                break
        
        if not obstacle:
            return json.dumps({
                "success": False,
                "error": f"No obstacle with id {obstacle_id} found"
            })
        
        obs_x, obs_y, obs_z = obstacle['x'], obstacle['y'], obstacle['z']
        rover_x = self.commander.rover_position[0]
        
        # Check if obstacle is behind the rover (already passed)
        if obs_x < rover_x - 2.0:
            self.get_logger().info(f"⏭️ Obstacle {obstacle_id} is behind the rover (obs_x={obs_x:.1f} < rover_x={rover_x:.1f}), skipping")
            self.current_obstacles.remove(obstacle)
            return json.dumps({
                "success": True,
                "message": f"Obstacle at ({obs_x}, {obs_y}, {obs_z}) is behind the rover — already passed. Removed from tracking.",
                "remaining_obstacles": len(self.current_obstacles)
            })
        
        radius = obstacle.get('radius', 3.0)
        offset = 2.0  # meters to swerve sideways
        
        # Swerve direction: left = -Z, right = +Z in Unity
        if direction == "right":
            avoid_z = obs_z + offset
        else:
            avoid_z = obs_z - offset
        
        avoid_x = obs_x
        avoid_y = obs_y
        
        # Travel heading (rover moves along +X axis between sites)
        travel_heading = 90.0  # degrees, facing +X in Unity
        
        self.get_logger().info(f"🚧 Navigating around obstacle: {obstacle['description']}")
        self.get_logger().info(f"   Direction: {direction}")
        self.get_logger().info(f"   Curve 1: swerve to ({avoid_x}, {avoid_y}, {avoid_z})")
        
        # === Curve 1: current position → avoidance point (swerve) ===
        self._publish_curved_goal(avoid_x, avoid_y, avoid_z, travel_heading, is_final=False)
        arrived1 = self.commander.wait_for_unity_arrival(timeout=30.0)
        
        if not arrived1:
            self.get_logger().warn("⚠️ Timeout on curve 1 (swerve), continuing anyway")
        
        # === Curve 2: avoidance point → rejoin original line past obstacle ===
        rejoin_x = obs_x + radius + 2.0  # Past the obstacle
        rejoin_z = obs_z  # Back on original line
        
        self.get_logger().info(f"   Curve 2: rejoin at ({rejoin_x}, {obs_y}, {rejoin_z})")
        self._publish_curved_goal(rejoin_x, obs_y, rejoin_z, travel_heading, is_final=True)
        arrived2 = self.commander.wait_for_unity_arrival(timeout=30.0)
        
        if not arrived2:
            self.get_logger().warn("⚠️ Timeout on curve 2 (rejoin), continuing anyway")
        
        # Mark obstacle as cleared
        self.current_obstacles.remove(obstacle)
        
        return json.dumps({
            "success": True,
            "message": f"Successfully navigated around the obstacle at ({obs_x}, {obs_y}, {obs_z}) going {direction}. Path is now clear.",
            "direction": direction,
            "avoidance_point": [avoid_x, avoid_y, avoid_z],
            "rejoin_point": [rejoin_x, obs_y, rejoin_z],
            "remaining_obstacles": len(self.current_obstacles)
        })
    
    def _publish_curved_goal(self, x, y, z, end_heading, is_final=False):
        """Publish a curved goal to Unity via /rover/curved_goal.
        
        Pose fields:
          position.x/y/z  = end point (Unity world coords)
          orientation.z    = end heading in degrees
          orientation.w    = 1.0 if final curve in sequence, 0.0 otherwise
        """
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.z = float(end_heading)
        msg.orientation.w = 1.0 if is_final else 0.0
        self.curved_goal_pub.publish(msg)
        self.get_logger().info(
            f"📤 Published curved goal: ({x}, {y}, {z}), "
            f"heading={end_heading}°, final={is_final}"
        )
    
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Route tool calls to implementations."""
        if tool_name == "deploy_antenna":
            return self.tool_deploy_antenna(arguments["site_id"])
        elif tool_name == "get_mission_status":
            return self.tool_get_mission_status()
        elif tool_name == "mission_complete":
            return self.tool_mission_complete(arguments.get("summary", ""))
        elif tool_name == "go_around_obstacle":
            return self.tool_go_around_obstacle(arguments["obstacle_id"], arguments.get("direction", "left"))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    # ========================================================================
    # REACT LOOP
    # ========================================================================
    
    def run_react_loop(self):
        """Main ReAct loop: prompt LLM -> execute tool -> repeat."""
        self.mission_active = True
        self.conversation_history = []
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("🚀 STARTING LLM-GUIDED MISSION")
        self.get_logger().info("="*60 + "\n")
        
        # Initial user message to kick off the mission
        user_message = "Begin the antenna deployment mission. Deploy all 4 sites in order."
        self.conversation_history.append({"role": "user", "content": user_message})
        
        iteration = 0
        max_iterations = 20  # Safety limit
        
        while self.mission_active and iteration < max_iterations:
            iteration += 1
            self.get_logger().info(f"\n--- ReAct Iteration {iteration} ---")
            
            # Call LLM with current conversation + tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.build_system_prompt()},
                    *self.conversation_history
                ],
                tools=TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Log the LLM's thinking (if any)
            if assistant_message.content:
                self.get_logger().info(f"\n{'─'*40}")
                self.get_logger().info(f"🧠 LLM REASONING:")
                self.get_logger().info(f"{assistant_message.content}")
                self.get_logger().info(f"{'─'*40}")
            else:
                self.get_logger().info(f"🧠 LLM: (no reasoning provided, direct tool call)")
            
            # Check if LLM wants to call tools
            if assistant_message.tool_calls:
                # Add assistant message to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    self.get_logger().info(f"🔧 Executing: {tool_name}({arguments})")
                    
                    result = self.execute_tool(tool_name, arguments)
                    
                    self.get_logger().info(f"📤 Result: {result}")
                    
                    # Add tool result to conversation
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                    
                    # Check if mission_complete was called
                    if tool_name == "mission_complete":
                        self.get_logger().info("\n" + "="*60)
                        self.get_logger().info("🎉 MISSION COMPLETE")
                        self.get_logger().info("="*60 + "\n")
                        self.save_conversation_log()
                        return
            else:
                # No tool calls - LLM is just responding
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                
                # If no tool call and no explicit completion, nudge the LLM
                if len(self.deployed_sites) < len(DEPLOYMENT_SITES):
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Continue with the next deployment."
                    })
        
        self.get_logger().warn(f"⚠️ ReAct loop ended after {iteration} iterations")
        self.save_conversation_log()
    
    def save_conversation_log(self):
        """Save the full conversation history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"mission_{timestamp}.json")
        
        log_data = {
            "timestamp": timestamp,
            "model": self.model,
            "deployed_sites": self.deployed_sites,
            "obstacles": self.current_obstacles,
            "system_prompt": self.build_system_prompt(),
            "conversation": self.conversation_history
        }
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        self.get_logger().info(f"\n📝 Conversation saved to: {log_file}")
        self.get_logger().info(f"   View with: cat {log_file} | python -m json.tool")


def main(args=None):
    rclpy.init(args=args)
    
    orchestrator = LLMOrchestrator()
    
    # Create executor for both nodes
    executor = MultiThreadedExecutor()
    executor.add_node(orchestrator)
    executor.add_node(orchestrator.commander)
    
    # Run ReAct loop in background thread
    def start_mission():
        time.sleep(3.0)  # Allow ROS connections to establish
        orchestrator.run_react_loop()
        # Shutdown after mission
        rclpy.shutdown()
    
    mission_thread = threading.Thread(target=start_mission, daemon=True)
    mission_thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.destroy_node()
        orchestrator.commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
