#!/usr/bin/env python3
"""
draw_background_bt.py

Renders a simple behavior tree diagram for the thesis background section.
Illustrates a Sequence → Fallback structure:

    Mission (Sequence)
      ├── Drive to Start          [Action]
      ├── Deploy antenna?  (Fallback)
      │     ├── Deploy antenna    [Action]
      │     └── Drive to End      [Action]  ← taken only if deploy fails
      └── Drive to End            [Action]  ← always runs

The Fallback always returns SUCCESS to the parent Sequence, so
"Drive to End" at the Sequence level runs regardless of deployment outcome.

Outputs:
    background_bt.png  (Graphviz dot render, same API as bt_orchestrator)
"""

import os
import py_trees


# ── Generic leaf node ─────────────────────────────────────────────────────────

class ActionNode(py_trees.behaviour.Behaviour):
    """Reusable stub action that always returns SUCCESS."""

    def __init__(self, name: str):
        super().__init__(name)

    def update(self):
        return py_trees.common.Status.SUCCESS


# ── Build tree ───────────────────────────────────────────────────────────────

def build_background_bt() -> py_trees.behaviour.Behaviour:
    """
    Mission (Sequence, memory=True)
      ├── Drive to Start
      ├── Deploy antenna? (Selector/Fallback, memory=True)
      │     ├── Deploy antenna
      │     └── Drive to End       ← fallback if deploy fails
      └── Drive to End             ← always runs
    """
    # Fallback: try to deploy; if it fails, drive straight to end.
    # memory=True avoids the asterisk (*) py_trees appends to memory=False
    # Selectors in the dot render.
    deploy_fallback = py_trees.composites.Selector(
        name="Deploy antenna?", memory=True
    )
    deploy_fallback.add_children([
        ActionNode("Deploy antenna"),
        ActionNode("Drive to end"),    # fallback branch
    ])

    # Root sequence: drive to start → deploy (or skip) → always drive to end.
    # The two "Drive to End" leaves need distinct py_trees names to avoid the
    # duplicate-ID asterisk; a single trailing space is invisible in the render.
    root = py_trees.composites.Sequence(
        name="Mission", memory=True
    )
    root.add_children([
        ActionNode("Drive to start"),
        deploy_fallback,
        ActionNode("Drive to end "),   # trailing space = unique ID, invisible in PNG
    ])

    return root


# ── Render ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = build_background_bt()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    py_trees.display.render_dot_tree(
        root,
        name="background_bt",
        target_directory=out_dir,
    )
    print(f"✅  Saved:  {os.path.join(out_dir, 'background_bt.png')}")
    print(f"✅  Dot:    {os.path.join(out_dir, 'background_bt.dot')}")
