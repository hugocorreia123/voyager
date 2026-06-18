"""Single-shot topology: the model answers directly, no tools.

This is the baseline every other topology is measured against. If a topology
with tools can't beat single-shot on a category, that's a finding.
"""
from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from voyager.config import get_llm
from voyager.topologies.base import RunResult, Topology

SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant. Answer the question as "
    "accurately and completely as you can from your own knowledge. If you are "
    "unsure, say so rather than inventing details."
)


class SingleShotTopology(Topology):
    name = "single_shot"

    def __init__(self) -> None:
        self.llm = get_llm()  # no tools bound — that is the point

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        resp = self.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
        )
        latency = time.perf_counter() - t0
        usage = getattr(resp, "usage_metadata", None) or {}
        return RunResult(
            topology=self.name,
            question=question,
            answer=str(resp.content),
            steps=1,
            tool_calls=[],
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_s=latency,
            trace=["SystemMessage", "HumanMessage", "AIMessage"],
        )
