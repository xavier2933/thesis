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
from driving_package.rover_commander import RoverCommander
import json
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
            "description": "Navigate the rover around a detected obstacle. Use this when an obstacle is blocking the path to the next deployment site. The rover will autonomously find a safe path around the obstacle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obstacle_id": {
                        "type": "integer",
                        "description": "The ID of the obstacle to navigate around"
                    }
                },
                "required": ["obstacle_id"]
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

AVAILABLE SITES:
{sites_info}

CURRENT OBSTACLES:
{obstacles_info}

RULES:
- Deploy sites in order (1 -> 2 -> 3 -> 4) unless blocked by obstacles
- After each deployment, check the mission status
- If an obstacle is detected, you may need to adjust the plan
- Call mission_complete when all sites are deployed

Think step by step. After each action, evaluate the result and decide the next step."""


class LLMOrchestrator(Node):
    """ReAct-style LLM planner for rover deployment."""
    
    def __init__(self):
        super().__init__('llm_orchestrator')
        
        # Initialize OpenAI client
        self.client = OpenAI()
        self.model = "gpt-5-nano"  # Can change to gpt-4o-mini for faster/cheaper
        
        # Initialize RoverCommander
        self.commander = RoverCommander()
        self.commander.sequence_started = True  # Prevent auto-start
        
        # Mission state
        self.deployed_sites = []
        self.current_obstacles = []  # Future: populated by obstacle detection
        self.mission_active = False
        self.conversation_history = []
        
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file = None
        
        self.get_logger().info("🤖 LLM Orchestrator initialized")
        self.get_logger().info(f"📁 Logs will be saved to: {LOG_DIR}")
        
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
            f"- Obstacle at ({o['x']}, {o['y']}, {o['z']}) with radius {o['radius']}m"
            for o in self.current_obstacles
        ])
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with current state."""
        return SYSTEM_PROMPT.format(
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
        
        # Execute deployment via RoverCommander
        try:
            antennas_deployed = self.commander.deploy_grid(waypoint_list)
            
            if antennas_deployed > 0:
                self.deployed_sites.append(site_id)
                
                # INJECT OBSTACLE after site 1 is deployed
                if site_id == 1 and not self.current_obstacles:
                    self.current_obstacles.append({
                        "id": 1,
                        "x": 422.0,  # Between site 1 end and site 2 start
                        "y": 18.0,
                        "z": 255.0,
                        "radius": 3.0,
                        "description": "Large rock blocking path to Site 2"
                    })
                    self.get_logger().info("\n" + "!"*50)
                    self.get_logger().info("🪨 OBSTACLE DETECTED: Large rock at (422, 18, 255)!")
                    self.get_logger().info("!"*50 + "\n")
                
                return json.dumps({
                    "success": True,
                    "message": f"Site {site_id} deployed successfully. {antennas_deployed} antenna(s) placed.",
                    "deployed_sites": self.deployed_sites,
                    "warning": "OBSTACLE DETECTED ahead! A large rock is blocking the path to Site 2. You must navigate around it before continuing." if site_id == 1 and self.current_obstacles else None
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Deployment at site {site_id} failed - no antennas placed"
                })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Deployment error: {str(e)}"
            })
    
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
    
    def tool_go_around_obstacle(self, obstacle_id: int) -> str:
        """Simulate navigating around an obstacle."""
        self.get_logger().info(f"🪨 LLM requested: go_around_obstacle(obstacle_id={obstacle_id})")
        
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
        
        # Simulate going around (does nothing for now)
        self.get_logger().info(f"🚧 Navigating around obstacle: {obstacle['description']}")
        time.sleep(1.0)  # Simulate some navigation time
        
        # Mark obstacle as cleared
        self.current_obstacles.remove(obstacle)
        
        return json.dumps({
            "success": True,
            "message": f"Successfully navigated around the obstacle at ({obstacle['x']}, {obstacle['y']}, {obstacle['z']}). Path is now clear.",
            "remaining_obstacles": len(self.current_obstacles)
        })
    
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Route tool calls to implementations."""
        if tool_name == "deploy_antenna":
            return self.tool_deploy_antenna(arguments["site_id"])
        elif tool_name == "get_mission_status":
            return self.tool_get_mission_status()
        elif tool_name == "mission_complete":
            return self.tool_mission_complete(arguments.get("summary", ""))
        elif tool_name == "go_around_obstacle":
            return self.tool_go_around_obstacle(arguments["obstacle_id"])
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
