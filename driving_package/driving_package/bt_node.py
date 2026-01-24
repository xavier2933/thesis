#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import py_trees
import time


# ------------------------
# Stub Behavior Nodes
# ------------------------

class SetAntennaIndex(py_trees.behaviour.Behaviour):
    def __init__(self, counter):
        super().__init__("SetAntennaIndex")
        self.counter = counter

    def update(self):
        self.logger.info("[INIT] Setting antenna index to 0")
        self.counter["index"] = 0
        return py_trees.common.Status.SUCCESS


class PlaceAntenna(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("PlaceAntenna")
        self.start_time = None
        self.duration = 2.0  # seconds

    def initialise(self):
        self.start_time = time.time()
        self.logger.info("[BT] Starting antenna placement")

    def update(self):
        elapsed = time.time() - self.start_time

        if elapsed < self.duration:
            self.logger.info(f"[BT] Placing antenna... ({elapsed:.1f}s)")
            return py_trees.common.Status.RUNNING

        self.logger.info("[BT] Antenna placed")
        return py_trees.common.Status.SUCCESS



class GoToNextDeploymentSite(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("GoToNextDeploymentSite")

    def update(self):
        self.logger.info("[ACTION] Driving to next deployment site (stub)")
        time.sleep(0.5)
        self.logger.info("[ACTION] Arrived at next deployment site")
        return py_trees.common.Status.SUCCESS


class IncrementAntennaIndex(py_trees.behaviour.Behaviour):
    def __init__(self, counter):
        super().__init__("IncrementAntennaIndex")
        self.counter = counter

    def update(self):
        self.counter["index"] += 1
        self.logger.info(
            f"[STATE] Incremented antenna index → {self.counter['index']}"
        )
        return py_trees.common.Status.SUCCESS


class CheckMoreAntennas(py_trees.behaviour.Behaviour):
    def __init__(self, counter):
        super().__init__("CheckMoreAntennas")
        self.counter = counter

    def update(self):
        idx = self.counter["index"]
        if idx < 4:
            self.logger.info(
                f"[CHECK] Antennas remaining (index={idx}) → CONTINUE"
            )
            return py_trees.common.Status.SUCCESS

        self.logger.info(
            f"[CHECK] No antennas remaining (index={idx}) → STOP"
        )
        return py_trees.common.Status.FAILURE


class ReportFinished(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("ReportFinished")

    def update(self):
        self.logger.info(
            "[MISSION] All antennas deployed. Mission complete."
        )
        return py_trees.common.Status.SUCCESS


# ------------------------
# BT Node
# ------------------------

class AntennaBTNode(Node):
    def __init__(self):
        super().__init__("driving_package")
        self.counter = {"index": 0}

        self.antenna_index = 0   # 👈 REQUIRED

        self.get_logger().info("[BT] Antenna Behavior Tree started")

        self.tree = self.create_tree()

        self.timer = self.create_timer(0.5, self.tick_tree)


    def create_tree(self):
        root = py_trees.composites.Sequence(
            "DeployAntennas", memory=True
        )

        root.add_children([
            SetAntennaIndex(self.counter),
            self.create_while_loop(),
            ReportFinished()
        ])

        return py_trees.trees.BehaviourTree(root)

    def create_while_loop(self):
        return py_trees.composites.Selector(
            name="WhileAntennasRemain",
            memory=True,
            children=[
                py_trees.composites.Sequence(
                    name="DeployOneAntenna",
                    memory=True,
                    children=[
                        CheckMoreAntennas(self.counter),
                        PlaceAntenna(),
                        IncrementAntennaIndex(self.counter),
                        py_trees.composites.Sequence(
                            name="DriveIfNotLast",
                            memory=True,
                            children=[
                                CheckMoreAntennas(self.counter),
                                GoToNextDeploymentSite()
                            ]
                        )
                    ]
                ),
                py_trees.behaviours.Success(name="ExitLoop")
            ]
        )

    def tick_tree(self):
        self.tree.tick()

        self.get_logger().info(
            f"[TICK] antenna_index={self.counter['index']} | "
            f"root_status={self.tree.root.status}"
        )

        if self.tree.root.status == py_trees.common.Status.SUCCESS:
            self.get_logger().info("[BT] Behavior tree completed — stopping ticks")
            self.timer.cancel()




def main():
    rclpy.init()
    node = AntennaBTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
