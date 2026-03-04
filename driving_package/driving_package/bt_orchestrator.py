#!/usr/bin/env python3
"""
Behavior-Tree Orchestrator for Rover Antenna Deployment

Uses py_trees to execute antenna deployments deterministically, with
per-waypoint granularity and reactive obstacle avoidance.

Usage:
    ros2 run driving_package bt_orchestrator
    ros2 run driving_package bt_orchestrator --ros-args -p debug_mode:=true
"""

import math
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
from driving_package.llm_orchestrator import DEPLOYMENT_SITES, DEPLOYMENT_ROWS, ROW_SPACING_Z


def compute_teleport_point(next_row_idx: int):
    """Return (x, y, z, heading) 2 m behind the first rope_start of next_row_idx.

    'Behind' means opposite to that row's travel direction, so the rover is
    already lined up and pointing the right way when it materialises.
    """
    row = DEPLOYMENT_ROWS[next_row_idx]
    d   = row["direction"]          # +1 for +X rows, -1 for -X rows
    next_sites = [s for s in DEPLOYMENT_SITES if s["row"] == next_row_idx]
    rs_x, rs_y, rs_z = next_sites[0]["waypoints"]["rope_start"]
    # Step 2 m back (opposite to travel direction)
    return rs_x - d * 2.0, rs_y, rs_z, row["heading"]


# ============================================================================
# BEHAVIOR NODES  — Navigation & Deployment
# ============================================================================

class GoToWaypoint(py_trees.behaviour.Behaviour):
    """Navigate the rover to a single waypoint (threaded).

    If *orchestrator* is supplied and the target waypoint is already behind
    the rover (in the current travel direction), navigation is skipped and
    SUCCESS is returned immediately — mirrors the LLM's adjusted-waypoint
    behaviour so the rover never reverses course after an obstacle avoidance.
    """

    def __init__(self, target: list, label: str,
                 commander: RoverCommander, logger, orchestrator=None):
        super().__init__(f"GoTo_{label}")
        self.target = target          # [x, y, z]
        self.commander = commander
        self._logger = logger
        self.orchestrator = orchestrator
        self._thread = None
        self._success = None
        self._skipped = False

    def initialise(self):
        self._success = None
        self._exception = None
        self._thread = None
        self._skipped = False
        x, y, z = self.target

        # Skip backward navigation: if the target is already behind the rover
        # in the travel direction, don't reverse — just treat it as arrived.
        if self.orchestrator is not None:
            d = self.orchestrator.travel_direction   # +1 or -1
            rover_x = self.commander.rover_position[0]
            if d * rover_x > d * x + 1.0:           # rover is > 1 m past target
                self._logger.info(
                    f"⏭️ BT: {self.name} target ({x:.1f}) is behind rover "
                    f"({rover_x:.1f}) — skipping backwards nav"
                )
                self._skipped = True
                self._success = True
                return

        self._logger.info(f"📍 BT: Navigating to {self.name} ({x}, {y}, {z})")
        self._thread = threading.Thread(target=self._go, daemon=True)
        self._thread.start()

    def update(self):
        if self._skipped:
            return py_trees.common.Status.SUCCESS
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            return py_trees.common.Status.SUCCESS
        if self._exception:
            self._logger.error(f"❌ BT: Exception reaching {self.name}: {self._exception}")
            return py_trees.common.Status.FAILURE
        # go_to_site returned False (Unity arrival timeout) — warn but continue.
        # The rover likely arrived; the arrival signal may have been dropped.
        self._logger.warn(
            f"⚠️ BT: Arrival timeout for {self.name} — assuming arrived, continuing"
        )
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        pass

    def _go(self):
        try:
            self._success = self.commander.go_to_site(*self.target)
        except Exception as e:
            self._logger.error(f"💥 BT: Exception in {self.name}: {e}")
            self._exception = e
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
# BEHAVIOR NODES  — Site reachability
# ============================================================================

class SiteReachable(py_trees.behaviour.Behaviour):
    """Condition: FAILURE if the rover has already passed the entire site.

    Mirrors the LLM's get_adjusted_site_waypoints() skip logic: if the rover
    is more than 1 m past rope_end (in the travel direction), the site cannot
    be deployed without reversing — mark as unreachable so the outer Sequence
    moves on to the next site.
    """

    def __init__(self, site: dict, orchestrator, logger):
        super().__init__(f"SiteReachable_S{site['site_id']}")
        self.site = site
        self.orchestrator = orchestrator
        self._logger = logger

    def update(self):
        d = self.orchestrator.travel_direction
        rover_x = self.orchestrator.commander.rover_position[0]
        rope_end_x = self.site["waypoints"]["rope_end"][0]
        if d * rover_x > d * rope_end_x + 1.0:
            self._logger.info(
                f"⏭️ BT: Rover ({rover_x:.1f}) past Site {self.site['site_id']} "
                f"rope_end ({rope_end_x:.1f}) — skipping site"
            )
            return py_trees.common.Status.FAILURE
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

    SWERVE_OFFSET = 3.0   # metres lateral
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
        if not self._success:
            self._logger.error("❌ BT: Obstacle avoidance maneuver failed")
        else:
            self._logger.info(
                "🚧 BT: Obstacle cleared — aborting current site, advancing to next"
            )
        # Always return FAILURE: avoidance is a detour, not a continuation.
        # The per-site Sequence sees FAILURE and terminates; the outer
        # mission Sequence (memory=True) then moves on to the next site.
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
            # Stop rope before maneuvering — mirrors abort_site() in the LLM
            # orchestrator.  Safe to call even if rope is already off.
            self.commander.set_rope(False)
            self._logger.info("🪢 BT: Rope stopped before obstacle avoidance")

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
    """Execute a semicircular U-turn to the next deployment row.

    Mirrors LLMOrchestrator.tool_turn_around() exactly:
      - Curve 1: current position → apex (mid-Z, 7.5 m ahead in travel dir)
      - Curve 2: apex → first rope_start of the next row
    Uses /rover/curved_goal (Pose) the same way the LLM version does.
    """

    def __init__(
        self,
        from_row_idx: int,
        orchestrator,          # BTOrchestrator — holds current_row/travel_dir
        curved_goal_pub,
        commander: RoverCommander,
        logger,
    ):
        super().__init__(f"TurnAround_R{from_row_idx}_to_R{from_row_idx+1}")
        self.from_row_idx = from_row_idx
        self.orchestrator = orchestrator
        self.curved_goal_pub = curved_goal_pub
        self.commander = commander
        self._logger = logger

        self._thread = None
        self._success = None

    def initialise(self):
        self._success = None
        self._thread = None
        self._thread = threading.Thread(target=self._turn, daemon=True)
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            self._logger.info(
                f"✅ BT: U-turn complete — now on Row {self.orchestrator.current_row}"
            )
            return py_trees.common.Status.SUCCESS
        self._logger.error("❌ BT: Turn-around failed")
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
        self._logger.info(
            f"📤 BT: Curved goal ({x:.1f}, {y:.1f}, {z:.1f}) "
            f"heading={heading}° final={is_final}"
        )

    def _turn(self):
        try:
            current_row_config = DEPLOYMENT_ROWS[self.from_row_idx]
            next_row_idx = self.from_row_idx + 1
            next_row_config = DEPLOYMENT_ROWS[next_row_idx]
            d = current_row_config["direction"]  # +1 or -1

            # First site of the next row → target landing point
            next_row_sites = [
                s for s in DEPLOYMENT_SITES if s["row"] == next_row_idx
            ]
            target_wp = next_row_sites[0]["waypoints"]["rope_start"]
            target_x, target_y, target_z = target_wp

            rover_pos = self.commander.rover_position
            rover_x, rover_y, rover_z = rover_pos[0], rover_pos[1], rover_pos[2]

            # ── Semicircular U-turn geometry (15 m diameter = row spacing) ──
            radius = ROW_SPACING_Z / 2.0          # 7.5 m
            mid_z  = (rover_z + target_z) / 2.0  # halfway between rows

            apex_x = rover_x + d * radius
            apex_y = rover_y
            # At the apex the rover faces perpendicular to its travel axis
            apex_heading = 0.0 if target_z > rover_z else 180.0

            current_heading = current_row_config["heading"]
            new_heading      = next_row_config["heading"]

            self._logger.info(
                f"🔄 BT: U-turn Row {self.from_row_idx} → Row {next_row_idx}"
            )
            self._logger.info(
                f"   Start:  ({rover_x:.1f}, {rover_y:.1f}, {rover_z:.1f}) "
                f"heading={current_heading}°"
            )
            self._logger.info(
                f"   Apex:   ({apex_x:.1f}, {apex_y:.1f}, {mid_z:.1f}) "
                f"heading={apex_heading}°"
            )
            self._logger.info(
                f"   Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) "
                f"heading={new_heading}°"
            )

            # Curve 1: start → apex
            self._publish_curved_goal(
                apex_x, apex_y, mid_z, apex_heading, is_final=False
            )
            arrived1 = self.commander.wait_for_unity_arrival(timeout=45.0)
            if not arrived1:
                self._logger.warn("⚠️ BT: Timeout on turn curve 1 (apex)")

            # Curve 2: apex → target
            self._publish_curved_goal(
                target_x, target_y, target_z, new_heading, is_final=True
            )
            arrived2 = self.commander.wait_for_unity_arrival(timeout=45.0)
            if not arrived2:
                self._logger.warn("⚠️ BT: Timeout on turn curve 2 (complete)")

            # Update orchestrator row state
            self.orchestrator.current_row      = next_row_idx
            self.orchestrator.travel_direction = next_row_config["direction"]

            self._success = True

        except Exception as e:
            self._logger.error(f"💥 BT: Exception during TurnAround: {e}")
            self._success = False


class TeleportTurnAround(py_trees.behaviour.Behaviour):
    """Instantly teleport the rover 2 m behind the first antenna of the next row.

    Publishes a single Pose to /rover/teleport.  Unity sets the Transform
    immediately and clears isNavigating, so wait_for_unity_arrival() returns
    almost instantly.
    """

    def __init__(
        self,
        from_row_idx: int,
        orchestrator,
        teleport_pub,
        commander: RoverCommander,
        logger,
    ):
        super().__init__(f"Teleport_R{from_row_idx}_to_R{from_row_idx+1}")
        self.from_row_idx  = from_row_idx
        self.orchestrator  = orchestrator
        self.teleport_pub  = teleport_pub
        self.commander     = commander
        self._logger       = logger
        self._thread       = None
        self._success      = None

    def initialise(self):
        self._success = None
        self._thread  = threading.Thread(target=self._do_teleport, daemon=True)
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING
        if self._success:
            self._logger.info(
                f"✅ BT: Teleport complete — now on Row {self.orchestrator.current_row}"
            )
            return py_trees.common.Status.SUCCESS
        self._logger.error("❌ BT: Teleport failed")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass

    def _do_teleport(self):
        try:
            next_row_idx = self.from_row_idx + 1
            x, y, z, heading = compute_teleport_point(next_row_idx)

            # Convert heading (code convention: 90°=+X, 270°=−X) to quaternion.
            #   90°  → identity        {x:0, y:0, z:0, w:1}
            #   270° → 180° yaw quat  {x:0, y:0, z:1, w:0}
            # Code heading convention (90°=+X, 270°=−X) is 90° off from Unity.
            # Subtract 90 so: heading=90 → unity 0° (identity), heading=270 → unity 180°.
            unity_degrees = heading + 180

            # 2. Convert to radians and DIVIDE BY 2 for Quaternions
            half_rad = math.radians(unity_degrees) / 2.0

            qz = math.sin(half_rad)
            qw = math.cos(half_rad)

            if abs(qz) < 1e-10: qz = 0.0
            if abs(qw) < 1e-10: qw = 0.0

            msg = Pose()
            msg.position.x    = float(x)
            msg.position.y    = float(z)   # Unity cross-track
            # Lift the rover slightly (20.5 instead of 20.0) 
            # to prevent ArticulationBody collision glitches on spawn
            msg.position.z    = 20.5       
            
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = float(qz)
            msg.orientation.w = float(qw)
            
            self.teleport_pub.publish(msg)

            self._logger.info(
                f"🚀 BT: Teleporting to Row {next_row_idx} → "
                f"({x:.1f}, {z:.1f}, 20.0) heading={heading}° (unity={heading-90:.0f}°) "
                f"→ quat z={qz:.4f} w={qw:.4f}"
            )

            # Wait for Unity to confirm (it sets isNavigating=false on receipt)
            arrived = self.commander.wait_for_unity_arrival(timeout=10.0)
            if not arrived:
                self._logger.warn("⚠️ BT: Teleport arrival timeout — assuming success")

            # Update orchestrator row state
            next_row_config = DEPLOYMENT_ROWS[next_row_idx]
            self.orchestrator.current_row      = next_row_idx
            self.orchestrator.travel_direction = next_row_config["direction"]
            self._success = True

        except Exception as e:
            self._logger.error(f"💥 BT: Exception during TeleportTurnAround: {e}")
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
        self.debug_mode = self.get_parameter("debug_mode").value
        self.declare_parameter("teleport_turnaround", False)
        self.teleport_turnaround = self.get_parameter("teleport_turnaround").value

        # RoverCommander
        self.commander = RoverCommander(debug_mode=self.debug_mode)
        self.commander.sequence_started = True  # prevent auto-start

        # Publishers
        self.site_pub = self.create_publisher(Int32, "/deployment_site_id", 10)
        self.curved_goal_pub = self.create_publisher(
            Pose, "/rover/curved_goal", 10
        )
        self.teleport_pub = self.create_publisher(
            Pose, "/rover/teleport", 10
        )

        # Obstacle tracking
        self.obstacles: list[dict] = []
        self._next_obstacle_id = 1
        self.create_subscription(
            String, "/rock_detection", self._rock_detection_cb, 10
        )

        # Deployment tracking
        self.deployed_sites: list[int] = []

        # Row tracking (mirrors LLMOrchestrator state)
        self.current_row      = 0
        self.travel_direction = DEPLOYMENT_ROWS[0]["direction"]  # +1

        # Build the tree
        self.tree = self._build_tree()

        # Tick timer
        self.tick_timer = self.create_timer(0.5, self._tick)

        # ── Print generated deployment sites for verification ──
        self.get_logger().info("🌳 BT Orchestrator initialized")
        self.get_logger().info(
            f"📋 Deployment plan: {len(DEPLOYMENT_SITES)} sites across "
            f"{len(DEPLOYMENT_ROWS)} rows"
        )
        for site in DEPLOYMENT_SITES:
            wp = site["waypoints"]
            self.get_logger().info(
                f"  Site {site['site_id']:>2} | Row {site['row']} | "
                f"rope_start={wp['rope_start']} "
                f"preamp={wp['preamp']} "
                f"rope_end={wp['rope_end']}"
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
        """Build the mission tree.

        Structure:
            DeployAllSites (Sequence, memory=True)
              ├── [Row-0 site subtrees]
              ├── TurnAround_R0_to_R1   ← semicircular U-turn
              ├── [Row-1 site subtrees]
              └── MissionComplete
        """
        root = py_trees.composites.Sequence(
            name="DeployAllSites", memory=True
        )

        # Group sites by row, preserving order
        rows = {}
        for site in DEPLOYMENT_SITES:
            rows.setdefault(site["row"], []).append(site)

        sorted_row_ids = sorted(rows.keys())
        for i, row_id in enumerate(sorted_row_ids):
            # Add all sites for this row
            for site in rows[row_id]:
                root.add_child(self._make_site_subtree(site))

            # After each row (except the last), insert a turn-around node
            if i < len(sorted_row_ids) - 1:
                if self.teleport_turnaround:
                    root.add_child(
                        TeleportTurnAround(
                            from_row_idx=row_id,
                            orchestrator=self,
                            teleport_pub=self.teleport_pub,
                            commander=self.commander,
                            logger=self.get_logger(),
                        )
                    )
                else:
                    root.add_child(
                        TurnAround(
                            from_row_idx=row_id,
                            orchestrator=self,
                            curved_goal_pub=self.curved_goal_pub,
                            commander=self.commander,
                            logger=self.get_logger(),
                        )
                    )

        root.add_child(
            MissionComplete(self.deployed_sites, self.get_logger())
        )
        # py_trees.display.render_dot_tree(root, name="bt_mission")

        return py_trees.trees.BehaviourTree(root)

    def _make_site_subtree(self, site: dict) -> py_trees.behaviour.Behaviour:
        """Build the per-site subtree with obstacle checks before each nav.

        The site sequence is wrapped in FailureIsSuccess so that an aborted
        site (obstacle avoidance mid-deployment) doesn't kill the mission —
        the outer Sequence simply advances to the next site.

        Structure per site:
            TrySite_N (FailureIsSuccess)
              └── Site_N (Sequence, memory=True)
                    ├── SiteReachable        ← skip if rover already past rope_end
                    ├── PublishSiteId
                    ├── ClearPath → GoToWaypoint(rope_start)  ← skips if behind rover
                    ├── StartRope
                    ├── ClearPath → GoToWaypoint(preamp)      ← skips if behind rover
                    ├── PickAndPlace
                    ├── ClearPath → GoToWaypoint(rope_end)    ← skips if behind rover
                    ├── StopRope
                    └── MarkDeployed
        """
        sid = site["site_id"]
        wp = site["waypoints"]
        log = self.get_logger()

        children = [
            SiteReachable(site, self, log),

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

        site_seq = py_trees.composites.Sequence(
            name=f"Site{sid}", memory=True, children=children,
        )

        # Wrap in FailureIsSuccess: if this site is aborted (obstacle) or
        # skipped (overshot), the outer mission Sequence sees SUCCESS and
        # continues to the next site rather than terminating the mission.
        return py_trees.decorators.FailureIsSuccess(
            name=f"TrySite{sid}", child=site_seq
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
                # Pass orchestrator so GoToWaypoint can skip backward nav
                # after an avoidance maneuver repositioned the rover.
                GoToWaypoint(waypoint, label, self.commander, log,
                             orchestrator=self),
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
