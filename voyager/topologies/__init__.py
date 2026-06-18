"""Agent topologies + a registry the CLI and eval harness select from.

Phase 1 COMPLETE: all five topologies registered.
"""
from voyager.topologies.multi_agent import MultiAgentTopology
from voyager.topologies.plan_execute import PlanExecuteTopology
from voyager.topologies.react import ReActTopology
from voyager.topologies.self_reflective import SelfReflectiveTopology
from voyager.topologies.single_shot import SingleShotTopology

TOPOLOGIES = {
    "single_shot": SingleShotTopology,
    "react": ReActTopology,
    "plan_execute": PlanExecuteTopology,
    "self_reflective": SelfReflectiveTopology,
    "multi_agent": MultiAgentTopology,
}

__all__ = [
    "ReActTopology", "SingleShotTopology", "PlanExecuteTopology",
    "SelfReflectiveTopology", "MultiAgentTopology", "TOPOLOGIES",
]
