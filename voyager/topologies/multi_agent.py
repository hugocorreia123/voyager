"""Multi-agent topology.

An orchestrator assigns focused sub-questions to specialist agents (a web
researcher and an academic researcher), each runs its own bounded tool loop, and
an aggregator synthesizes their findings. Differs from plan-execute by giving
each agent a distinct role rather than running generic sequential steps.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from voyager.config import get_llm
from voyager.tools import TOOLS
from voyager.topologies.base import RunResult, Topology

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SPECIALISTS = {
    "web_researcher": (
        "You are a web researcher. Find current, real-world facts using the "
        "web_search tool, then report concise findings."
    ),
    "academic_researcher": (
        "You are an academic researcher. Find scholarly and technical sources "
        "using the arxiv_search tool, then report concise findings."
    ),
}
AGG_PROMPT = (
    "You are the lead researcher. Synthesize the specialists' findings into a "
    "clear, well-supported final answer to the original question."
)


class _State(TypedDict):
    question: str
    assignments: dict
    findings: Annotated[list[str], operator.add]
    answer: str
    responses: Annotated[list, operator.add]
    tool_names: Annotated[list[str], operator.add]


class MultiAgentTopology(Topology):
    name = "multi_agent"

    def __init__(self, specialist_max_rounds: int = 3) -> None:
        self.specialist_max_rounds = specialist_max_rounds
        self.llm = get_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(_State)
        g.add_node("orchestrate", self._orchestrate)
        g.add_node("specialists", self._specialists)
        g.add_node("aggregate", self._aggregate)
        g.add_edge(START, "orchestrate")
        g.add_edge("orchestrate", "specialists")
        g.add_edge("specialists", "aggregate")
        g.add_edge("aggregate", END)
        return g.compile()

    def _orchestrate(self, state: _State) -> dict:
        roster = ", ".join(SPECIALISTS)
        prompt = (
            f"You are an orchestrator coordinating specialist agents ({roster}). "
            "For each specialist relevant to the question, write a focused "
            "sub-question. Output one line per specialist as 'role: sub-question'. "
            "Use only the listed role names."
        )
        resp = self.llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["question"])]
        )
        assignments = {}
        for line in str(resp.content).splitlines():
            if ":" in line:
                role, _, subq = line.partition(":")
                role = role.strip().lower()
                if role in SPECIALISTS and subq.strip():
                    assignments[role] = subq.strip()
        if not assignments:  # fallback: every specialist gets the question
            assignments = {r: state["question"] for r in SPECIALISTS}
        return {"assignments": assignments, "responses": [resp]}

    def _specialists(self, state: _State) -> dict:
        findings, all_responses, tools_used = [], [], []
        for role, subq in state["assignments"].items():
            msgs = [SystemMessage(content=SPECIALISTS[role]),
                    HumanMessage(content=subq)]
            last_content = ""
            for _ in range(self.specialist_max_rounds):
                resp = self.llm_tools.invoke(msgs)
                all_responses.append(resp)
                msgs.append(resp)
                last_content = str(resp.content)
                calls = getattr(resp, "tool_calls", None)
                if not calls:
                    break
                for call in calls:
                    tools_used.append(call["name"])
                    tool = _TOOLS_BY_NAME.get(call["name"])
                    try:
                        result = tool.invoke(call["args"]) if tool else "Unknown tool"
                    except Exception as e:
                        result = f"Tool error: {e}"
                    msgs.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
            findings.append(f"[{role}] {last_content}")
        return {"findings": findings, "responses": all_responses,
                "tool_names": tools_used}

    def _aggregate(self, state: _State) -> dict:
        joined = "\n\n".join(state["findings"])
        resp = self.llm.invoke(
            [SystemMessage(content=AGG_PROMPT),
             HumanMessage(content=f"Question: {state['question']}\n\nSpecialist findings:\n{joined}")]
        )
        return {"answer": str(resp.content), "responses": [resp]}

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        final = self.graph.invoke(
            {"question": question, "assignments": {}, "findings": [],
             "answer": "", "responses": [], "tool_names": []},
            {"recursion_limit": 50},
        )
        latency = time.perf_counter() - t0
        in_tok = out_tok = 0
        for r in final["responses"]:
            u = getattr(r, "usage_metadata", None) or {}
            in_tok += u.get("input_tokens", 0)
            out_tok += u.get("output_tokens", 0)
        return RunResult(
            topology=self.name,
            question=question,
            answer=final["answer"],
            steps=len(final["responses"]),
            tool_calls=final["tool_names"],
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            trace=[f"{r}: {q}" for r, q in final["assignments"].items()],
        )
