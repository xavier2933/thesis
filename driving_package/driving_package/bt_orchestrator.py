#!/usr/bin/env python3
"""
Behavior-Tree Orchestrator for Rover Antenna Deployment

Uses py_trees to execute antenna deployments deterministically, with
per-waypoint granularity and reactive obstacle avoidance.

Usage:
    ros2 run driving_package bt_orchestrator
    ros2 run driving_package bt_orchestrator --ros-args -p debug_mode:=true
"""

import re
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Pose
import py_trees
import threading
import time

from driving_package.rover_commander import RoverCommander
from driving_package.llm_orchestrator import DEPLOYMENT_SITES


# ============================================================================
# BEHAVIOR NODES  — Navigation & Deployment
# ============================================================================

class GoToWaypoint(py_trees.behaviour.Behaviour):
    """Navigate the rover to a single waypoint (threaded)."""

    def __init__(self, target: list, label: str,
                 commander: RoverCommander, logger):
        super().__init__(f"GoTo_{label}")
        self.target = target          # [x, y, z]
        self.commander = commander
        self._logger = logger
        self._thread = None
        self._success = None

    def initialise(self):
        self._success = None
        self._thread = None
        x, y, z = self.target
        self._logger.info(f"📍 BT: Navigating to {self.name} ({x}, {y}, {z})")
        self._thread = threading.Thread(target=self._go, daemon=True)
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            return py_trees.common.Status.SUCCESS
        self._logger.error(f"❌ BT: Failed to reach {self.name}")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass

    def _go(self):
        try:
            self._success = self.commander.go_to_site(*self.target)
        except Exception as e:
            self._logger.error(f"💥 BT: Exception in {self.name}: {e}")
            self._success = False


class StartRope(py_trees.behaviour.Behaviour):
    """Start rope deployment (instant)."""

    def __init__(self, commander: RoverCommander, logger):
        super().__init__("StartRope")
        self.commander = commander
        self._logger = logger

    def update(self):
        self._logger.info("🪢 BT: Starting rope deployment")
        self.commander.set_rope(True)
        return py_trees.common.Status.SUCCESS


class StopRope(py_trees.behaviour.Behaviour):
    """Stop rope deployment and publish placement_complete (instant)."""

    def __init__(self, site_id: int, commander: RoverCommander,
                 site_pub, logger):
        super().__init__(f"StopRope_S{site_id}")
        self.site_id = site_id
        self.commander = commander
        self.site_pub = site_pub
        self._logger = logger

    def update(self):
        # Publish placement_complete (same as deploy_grid's last-waypoint)
        msg = Int32()
        msg.data = self.site_id
        self.commander.placement_complete_pub.publish(msg)
        self._logger.info(
            f"📤 BT: Sent placement_complete for Site {self.site_id}"
        )
        self._logger.info("🪢 BT: Stopping rope deployment")
        self.commander.set_rope(False)
        return py_trees.common.Status.SUCCESS


class PickAndPlace(py_trees.behaviour.Behaviour):
    """Execute pick-and-place at the current waypoint (threaded)."""

    def __init__(self, commander: RoverCommander, debug_mode: bool, logger):
        super().__init__("PickAndPlace")
        self.commander = commander
        self.debug_mode = debug_mode
        self._logger = logger
        self._thread = None
        self._success = None

    def initialise(self):
        self._success = None
        self._thread = None
        if self.debug_mode:
            self._logger.info("⚡ BT: DEBUG — skipping pick & place")
            self._success = True
            return
        self._logger.info("🦾 BT: Executing pick & place")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            return py_trees.common.Status.SUCCESS
        self._logger.error("❌ BT: Pick & place failed")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass

    def _run(self):
        try:
            self._success = self.commander.deploy_antenna_at_current_site()
        except Exception as e:
            self._logger.error(f"💥 BT: Exception in PickAndPlace: {e}")
            self._success = False


class PublishSiteId(py_trees.behaviour.Behaviour):
    """Publish site_id to Unity so it knows which antenna to track."""

    def __init__(self, site_id: int, site_pub, logger):
        super().__init__(f"PublishSiteId_{site_id}")
        self.site_id = site_id
        self.site_pub = site_pub
        self._logger = logger

    def update(self):
        msg = Int32()
        msg.data = self.site_id
        self.site_pub.publish(msg)
        self._logger.info(
            f"🎯 BT: Starting deployment at site {self.site_id}"
        )
        return py_trees.common.Status.SUCCESS


# ============================================================================
# BEHAVIOR NODES  — Obstacle Avoidance
# ============================================================================

class ObstacleAhead(py_trees.behaviour.Behaviour):
    """Condition: SUCCESS if an obstacle is ahead of the rover.

    "Ahead" means obstacle_x > rover_x - 2.0  (same logic as the LLM
    orchestrator).  The node does NOT consume/remove the obstacle.
    """

    def __init__(self, obstacles: list, commander: RoverCommander, logger):
        super().__init__("ObstacleAhead?")
        self.obstacles = obstacles
        self.commander = commander
        self._logger = logger

    def update(self):
        rover_x = self.commander.rover_position[0]
        for obs in self.obstacles:
            if obs["x"] > rover_x - 2.0:
                self._logger.info(
                    f"🪨 BT: Obstacle id={obs['id']} ahead "
                    f"at ({obs['x']}, {obs['y']}, {obs['z']})"
                )
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class AvoidObstacle(py_trees.behaviour.Behaviour):
    """Navigate around the nearest ahead obstacle using two curved goals.

    Replicates the logic of LLMOrchestrator.tool_go_around_obstacle().
    Threaded — returns RUNNING while the manoeuvre is in progress.

    Never seen this spelling of maneuver
    """

    SWERVE_OFFSET = 5.0   # metres lateral
    TRAVEL_HEADING = 90.0  # degrees, +X axis

    def __init__(self, obstacles: list, commander: RoverCommander,
                 curved_goal_pub, logger):
        super().__init__("AvoidObstacle")
        self.obstacles = obstacles
        self.commander = commander
        self.curved_goal_pub = curved_goal_pub
        self._logger = logger

        self._thread = None
        self._success = None

    def initialise(self):
        self._success = None
        self._thread = None
        self._thread = threading.Thread(target=self._avoid, daemon=True)
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            return py_trees.common.Status.SUCCESS
        self._logger.error("❌ BT: Obstacle avoidance failed")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass

    def _publish_curved_goal(self, x, y, z, heading, is_final):
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.z = float(heading)
        msg.orientation.w = 1.0 if is_final else 0.0
        self.curved_goal_pub.publish(msg)

    def _avoid(self):
        try:
            rover_x = self.commander.rover_position[0]

            # Find nearest ahead obstacle
            ahead = [o for o in self.obstacles if o["x"] > rover_x - 2.0]
            if not ahead:
                self._success = True
                return
            obstacle = min(ahead, key=lambda o: o["x"])

            obs_x = obstacle["x"]
            obs_y = obstacle["y"]
            obs_z = obstacle["z"]
            radius = obstacle.get("radius", 3.0)

            # Always swerve left (-Z) — matches LLM default
            avoid_z = obs_z - self.SWERVE_OFFSET

            self._logger.info(
                f"🚧 BT: Avoiding obstacle id={obstacle['id']} "
                f"at ({obs_x}, {obs_y}, {obs_z}) — swerving left"
            )

            # Curve 1: swerve to avoidance point
            self._publish_curved_goal(
                obs_x, obs_y, avoid_z, self.TRAVEL_HEADING, is_final=False
            )
            self.commander.wait_for_unity_arrival(timeout=30.0)

            # Curve 2: rejoin original line past obstacle
            rejoin_x = obs_x + radius + 2.0
            self._publish_curved_goal(
                rejoin_x, obs_y, obs_z, self.TRAVEL_HEADING, is_final=True
            )
            self.commander.wait_for_unity_arrival(timeout=30.0)

            # Clear this obstacle
            self.obstacles.remove(obstacle)
            self._logger.info(
                f"✅ BT: Cleared obstacle id={obstacle['id']}"
            )
            self._success = True

        except Exception as e:
            self._logger.error(f"💥 BT: Exception in AvoidObstacle: {e}")
            self._success = False


# ============================================================================
# BEHAVIOR NODES  — Mission bookkeeping
# ============================================================================

class MissionComplete(py_trees.behaviour.Behaviour):
    """Leaf node that logs mission summary and always returns SUCCESS."""

    def __init__(self, deployed_sites: list, logger):
        super().__init__("MissionComplete")
        self.deployed_sites = deployed_sites
        self._logger = logger

    def update(self):
        total = len(DEPLOYMENT_SITES)
        done = len(self.deployed_sites)
        self._logger.info("\n" + "=" * 60)
        self._logger.info(
            f"🎉 MISSION COMPLETE — {done}/{total} sites deployed"
        )
        self._logger.info("=" * 60 + "\n")
        return py_trees.common.Status.SUCCESS


class TurnAround(py_trees.behaviour.Behaviour):
    """Navigate the rover to a specified return point (threaded)."""

    def __init__(self, target: list, commander: RoverCommander, logger):
        super().__init__("TurnAround")
        self.target = target
        self.commander = commander
        self._logger = logger

        self._thread = None
        self._success = None

    def initialise(self):
        self._success = None
        self._thread = None
        x, y, z = self.target
        self._logger.info(
            f"🔄 BT: Turning around — heading to ({x}, {y}, {z})"
        )
        self._thread = threading.Thread(
            target=self._navigate, daemon=True
        )
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            self._logger.info("✅ BT: Arrived at return point")
            return py_trees.common.Status.SUCCESS
        self._logger.error("❌ BT: Failed to reach return point")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass

    def _navigate(self):
        try:
            x, y, z = self.target
            self._success = self.commander.go_to_site(x, y, z)
        except Exception as e:
            self._logger.error(
                f"💥 BT: Exception during turn-around: {e}"
            )
            self._success = False


class _MarkDeployed(py_trees.behaviour.Behaviour):
    """Tiny bookkeeping node: appends a site_id to the deployed list."""

    def __init__(self, site_id: int, deployed_list: list):
        super().__init__(f"MarkDeployed_{site_id}")
        self.site_id = site_id
        self.deployed_list = deployed_list

    def update(self):
        if self.site_id not in self.deployed_list:
            self.deployed_list.append(self.site_id)
        return py_trees.common.Status.SUCCESS


# ============================================================================
# ROS 2 NODE
# ============================================================================

class BTOrchestrator(Node):
    """ROS node that builds and ticks a py_trees behavior tree."""

    def __init__(self):
        super().__init__("bt_orchestrator")

        # Parameters
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("return_point", [400.0, 18.0, 255.0])
        self.debug_mode = self.get_parameter("debug_mode").value
        self.return_point = (
            self.get_parameter("return_point")
            .get_parameter_value()
            .double_array_value
        )

        # RoverCommander
        self.commander = RoverCommander(debug_mode=self.debug_mode)
        self.commander.sequence_started = True  # prevent auto-start

        # Publishers
        self.site_pub = self.create_publisher(Int32, "/deployment_site_id", 10)
        self.curved_goal_pub = self.create_publisher(
            Pose, "/rover/curved_goal", 10
        )

        # Obstacle tracking
        self.obstacles: list[dict] = []
        self._next_obstacle_id = 1
        self.create_subscription(
            String, "/rock_detection", self._rock_detection_cb, 10
        )

        # Deployment tracking
        self.deployed_sites: list[int] = []

        # Build the tree
        self.tree = self._build_tree()

        # Tick timer
        self.tick_timer = self.create_timer(0.5, self._tick)

        self.get_logger().info("🌳 BT Orchestrator initialized")
        self.get_logger().info(
            f"📍 Return point: {list(self.return_point)}"
        )
        if self.debug_mode:
            self.get_logger().info(
                "⚡ DEBUG MODE: arm operations skipped in RoverCommander"
            )

    # -- obstacle detection --------------------------------------------------

    def _rock_detection_cb(self, msg: String):
        """Parse rock position from Unity (same as LLMOrchestrator)."""
        try:
            match = re.search(
                r'\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', msg.data
            )
            if not match:
                self.get_logger().warn(
                    f"Could not parse rock from: {msg.data}"
                )
                return

            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))

            # De-duplicate (within 2 m)
            for o in self.obstacles:
                dist = (
                    (o["x"] - x) ** 2
                    + (o["y"] - y) ** 2
                    + (o["z"] - z) ** 2
                ) ** 0.5
                if dist < 2.0:
                    return

            obstacle = {
                "id": self._next_obstacle_id,
                "x": x, "y": y, "z": z,
                "radius": 3.0,
                "description": f"Rock at ({x}, {y}, {z})",
            }
            self._next_obstacle_id += 1
            self.obstacles.append(obstacle)
            self.get_logger().info(
                f"🪨 Rock detected! id={obstacle['id']} "
                f"at ({x}, {y}, {z})"
            )
        except Exception as e:
            self.get_logger().error(f"Rock parse error: {e}")

    # -- tree construction ---------------------------------------------------

    def _build_tree(self) -> py_trees.trees.BehaviourTree:
        root = py_trees.composites.Sequence(
            name="DeployAllSites", memory=True
        )

        for site in DEPLOYMENT_SITES:
            root.add_child(self._make_site_subtree(site))

        root.add_child(
            MissionComplete(self.deployed_sites, self.get_logger())
        )
        root.add_child(
            TurnAround(
                target=list(self.return_point),
                commander=self.commander,
                logger=self.get_logger(),
            )
        )

        return py_trees.trees.BehaviourTree(root)

    def _make_site_subtree(self, site: dict) -> py_trees.behaviour.Behaviour:
        """Build the per-site subtree with obstacle checks before each nav.

        Structure per site:
            Site_N (Sequence, memory=True)
              ├── PublishSiteId
              ├── ClearPath → GoToWaypoint(rope_start)
              ├── StartRope
              ├── ClearPath → GoToWaypoint(preamp)
              ├── PickAndPlace
              ├── ClearPath → GoToWaypoint(rope_end)
              ├── StopRope
              └── MarkDeployed
        """
        sid = site["site_id"]
        wp = site["waypoints"]
        log = self.get_logger()

        children = [
            PublishSiteId(sid, self.site_pub, log),

            # --- rope_start ---
            self._make_nav_with_obstacle_check(
                wp["rope_start"], f"S{sid}_RopeStart"
            ),
            StartRope(self.commander, log),

            # --- preamp ---
            self._make_nav_with_obstacle_check(
                wp["preamp"], f"S{sid}_Preamp"
            ),
            PickAndPlace(self.commander, self.debug_mode, log),

            # --- rope_end ---
            self._make_nav_with_obstacle_check(
                wp["rope_end"], f"S{sid}_RopeEnd"
            ),
            StopRope(sid, self.commander, self.site_pub, log),

            _MarkDeployed(sid, self.deployed_sites),
        ]

        return py_trees.composites.Sequence(
            name=f"Site{sid}", memory=True, children=children,
        )

    def _make_nav_with_obstacle_check(
        self, waypoint: list, label: str
    ) -> py_trees.behaviour.Behaviour:
        """Wrap a GoToWaypoint in a pre-navigation obstacle check.

        Structure:
            NavTo_<label> (Sequence, memory=True)
              ├── ClearPath (Selector, memory=True)
              │     ├── Inverter(ObstacleAhead?)   ← path clear → SUCCESS
              │     └── AvoidObstacle              ← path blocked → avoid
              └── GoToWaypoint(<label>)
        """
        log = self.get_logger()

        clear_path = py_trees.composites.Selector(
            name=f"ClearPath_{label}",
            memory=True,
            children=[
                py_trees.decorators.Inverter(
                    name=f"NothingAhead_{label}",
                    child=ObstacleAhead(
                        self.obstacles, self.commander, log
                    ),
                ),
                AvoidObstacle(
                    self.obstacles, self.commander,
                    self.curved_goal_pub, log,
                ),
            ],
        )

        return py_trees.composites.Sequence(
            name=f"NavTo_{label}",
            memory=True,
            children=[
                clear_path,
                GoToWaypoint(waypoint, label, self.commander, log),
            ],
        )

    # -- tick callback -------------------------------------------------------

    def _tick(self):
        self.tree.tick()

        status = self.tree.root.status
        self.get_logger().info(
            f"[TICK] deployed={self.deployed_sites} "
            f"obstacles={len(self.obstacles)} | root={status}"
        )

        if status in (
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        ):
            self.get_logger().info("🌳 BT finished — stopping ticks")
            self.tick_timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    orchestrator = BTOrchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(orchestrator)
    executor.add_node(orchestrator.commander)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.destroy_node()
        orchestrator.commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
