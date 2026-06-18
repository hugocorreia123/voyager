"""Agent topologies + a registry the CLI and eval harness select from.

Phase 1 progress: single_shot, react, plan_execute, self_reflective.
Still to come: multi_agent.
"""
from voyager.topologies.plan_execute import PlanExecuteTopology
from voyager.topologies.react import ReActTopology
from voyager.topologies.self_reflective import SelfReflectiveTopology
from voyager.topologies.single_shot import SingleShotTopology

TOPOLOGIES = {
    "single_shot": SingleShotTopology,
    "react": ReActTopology,
    "plan_execute": PlanExecuteTopology,
    "self_reflective": SelfReflectiveTopology,
}

__all__ = [
    "ReActTopology", "SingleShotTopology", "PlanExecuteTopology",
    "SelfReflectiveTopology", "TOPOLOGIES",
]
