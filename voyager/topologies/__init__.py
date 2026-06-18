"""Agent topologies + a registry the CLI and eval harness select from.

Phase 1 progress: single_shot + react. Still to come: plan_execute,
multi_agent, self_reflective.
"""
from voyager.topologies.react import ReActTopology
from voyager.topologies.single_shot import SingleShotTopology

TOPOLOGIES = {
    "single_shot": SingleShotTopology,
    "react": ReActTopology,
}

__all__ = ["ReActTopology", "SingleShotTopology", "TOPOLOGIES"]
