#!/usr/bin/env python3
"""
Behavior-Tree Orchestrator for Rover Antenna Deployment

Uses py_trees to execute the same deploy_grid() sequence that
llm_orchestrator.py drives via LLM tool calls, but deterministically.

Usage:
    ros2 run driving_package bt_orchestrator
    ros2 run driving_package bt_orchestrator --ros-args -p debug_mode:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Int32
import py_trees
import threading
import time

from driving_package.rover_commander import RoverCommander
from driving_package.llm_orchestrator import DEPLOYMENT_SITES


# ============================================================================
# BEHAVIOR NODES
# ============================================================================

class DeploySite(py_trees.behaviour.Behaviour):
    """Deploy a single antenna site via RoverCommander.deploy_grid().

    On first tick the blocking call is dispatched to a background thread;
    subsequent ticks return RUNNING until the thread finishes, then
    SUCCESS or FAILURE depending on the result.
    """

    def __init__(self, site_id: int, commander: RoverCommander,
                 site_pub, logger):
        super().__init__(f"DeploySite_{site_id}")
        self.site_id = site_id
        self.commander = commander
        self.site_pub = site_pub
        self._logger = logger

        # Look up waypoints from shared config
        site = DEPLOYMENT_SITES[site_id - 1]
        wp = site["waypoints"]
        self.waypoints = [wp["rope_start"], wp["preamp"], wp["rope_end"]]

        # Thread bookkeeping
        self._thread = None
        self._result = None  # None = not started, int = antennas deployed

    # -- py_trees lifecycle --------------------------------------------------

    def initialise(self):
        """Called once when a parent composite first ticks this node."""
        self._result = None
        self._thread = None
        self._logger.info(f"🎯 BT: Starting deployment at site {self.site_id}")

        # Tell Unity which site we're deploying
        msg = Int32()
        msg.data = self.site_id
        self.site_pub.publish(msg)

        # Kick off the blocking work on a daemon thread
        self._thread = threading.Thread(
            target=self._deploy, daemon=True
        )
        self._thread.start()

    def update(self):
        if self._thread is not None and self._thread.is_alive():
            return py_trees.common.Status.RUNNING

        # Thread finished (or was never started – shouldn't happen)
        if self._result is not None and self._result > 0:
            self._logger.info(
                f"✅ BT: Site {self.site_id} deployed "
                f"({self._result} antenna(s))"
            )
            return py_trees.common.Status.SUCCESS

        self._logger.error(f"❌ BT: Site {self.site_id} failed")
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        # Nothing to clean up; the thread is a daemon and will die with
        # the process if the tree is torn down.
        pass

    # -- internals -----------------------------------------------------------

    def _deploy(self):
        """Run deploy_grid (blocking) in background thread."""
        try:
            self._result = self.commander.deploy_grid(
                self.waypoints, site_id=self.site_id
            )
        except Exception as e:
            self._logger.error(
                f"💥 BT: Exception during site {self.site_id}: {e}"
            )
            self._result = 0


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


# ============================================================================
# ROS 2 NODE
# ============================================================================

class BTOrchestrator(Node):
    """ROS node that builds and ticks a py_trees behavior tree."""

    def __init__(self):
        super().__init__("bt_orchestrator")

        # Parameters
        self.declare_parameter("debug_mode", False)
        debug_mode = self.get_parameter("debug_mode").value

        # RoverCommander (same pattern as orchestrate_deployment.py)
        self.commander = RoverCommander(debug_mode=debug_mode)
        self.commander.sequence_started = True  # prevent auto-start

        # Publisher shared with leaf nodes
        self.site_pub = self.create_publisher(Int32, "/deployment_site_id", 10)

        # Track which sites succeed (populated by DeploySite nodes)
        self.deployed_sites: list[int] = []

        # Build the tree
        self.tree = self._build_tree()

        # Tick on a timer so the ROS executor stays responsive
        self.tick_timer = self.create_timer(0.5, self._tick)

        self.get_logger().info("🌳 BT Orchestrator initialized")
        if debug_mode:
            self.get_logger().info(
                "⚡ DEBUG MODE: arm operations skipped in RoverCommander"
            )

    # -- tree construction ---------------------------------------------------

    def _build_tree(self) -> py_trees.trees.BehaviourTree:
        root = py_trees.composites.Sequence(
            name="DeployAllSites", memory=True
        )

        for site in DEPLOYMENT_SITES:
            site_node = DeploySite(
                site_id=site["site_id"],
                commander=self.commander,
                site_pub=self.site_pub,
                logger=self.get_logger(),
            )
            # Wrap each in a tiny sequence so we can track success
            wrapper = py_trees.composites.Sequence(
                name=f"Site{site['site_id']}", memory=True,
                children=[
                    site_node,
                    _MarkDeployed(site["site_id"], self.deployed_sites),
                ],
            )
            root.add_child(wrapper)

        root.add_child(
            MissionComplete(self.deployed_sites, self.get_logger())
        )

        return py_trees.trees.BehaviourTree(root)

    # -- tick callback -------------------------------------------------------

    def _tick(self):
        self.tree.tick()

        status = self.tree.root.status
        self.get_logger().info(
            f"[TICK] deployed={self.deployed_sites} | "
            f"root={status}"
        )

        if status in (
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        ):
            self.get_logger().info("🌳 BT finished — stopping ticks")
            self.tick_timer.cancel()


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
# ENTRY POINT
# ============================================================================

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
