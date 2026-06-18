"""Agent topologies + a registry the CLI and eval harness select from.

Phase 1 progress: single_shot, react, plan_execute.
Still to come: multi_agent, self_reflective.
"""
from voyager.topologies.plan_execute import PlanExecuteTopology
from voyager.topologies.react import ReActTopology
from voyager.topologies.single_shot import SingleShotTopology

TOPOLOGIES = {
    "single_shot": SingleShotTopology,
    "react": ReActTopology,
    "plan_execute": PlanExecuteTopology,
}

__all__ = [
    "ReActTopology", "SingleShotTopology", "PlanExecuteTopology", "TOPOLOGIES",
]
