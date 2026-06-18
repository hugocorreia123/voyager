"""Agent topologies. Phase 0 ships ReAct; Phases 1 adds the other four
(single-shot, plan-and-execute, multi-agent, self-reflective)."""
from voyager.topologies.react import ReActTopology

__all__ = ["ReActTopology"]
