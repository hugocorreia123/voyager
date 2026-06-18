"""Self-reflective topology.

Produce a draft (ReAct-style, with tools), then have a critic review it for
completeness/accuracy/grounding. If the critic isn't satisfied, revise — up to a
capped number of reflection rounds. Trades extra cost for higher answer quality.
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

ACT_PROMPT = (
    "You are a rigorous research assistant. Use the tools to gather evidence, "
    "then give a clear, well-supported answer to the question."
)
CRITIC_PROMPT = (
    "You are a strict reviewer. Assess whether the answer is complete, accurate, "
    "and well-grounded for the question. If it is good enough, reply with exactly "
    "'ACCEPT'. Otherwise reply 'REVISE:' followed by the specific gaps or errors "
    "to fix."
)


class _State(TypedDict):
    question: str
    draft: str
    critique: str
    reflections: int
    answer: str
    responses: Annotated[list, operator.add]
    tool_names: Annotated[list[str], operator.add]


class SelfReflectiveTopology(Topology):
    name = "self_reflective"

    def __init__(self, max_reflections: int = 2, max_act_rounds: int = 5) -> None:
        self.max_reflections = max_reflections
        self.max_act_rounds = max_act_rounds
        self.llm = get_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(_State)
        g.add_node("act", self._act)
        g.add_node("reflect", self._reflect)
        g.add_edge(START, "act")
        g.add_edge("act", "reflect")
        g.add_conditional_edges(
            "reflect", self._route, {"revise": "act", "accept": END}
        )
        return g.compile()

    def _act(self, state: _State) -> dict:
        guidance = ""
        if state.get("critique"):
            guidance = (
                "\n\nA reviewer raised these issues with your previous answer; "
                f"address them:\n{state['critique']}"
            )
        msgs = [
            SystemMessage(content=ACT_PROMPT),
            HumanMessage(content=state["question"] + guidance),
        ]
        responses, tools_used = [], []
        for _ in range(self.max_act_rounds):
            resp = self.llm_tools.invoke(msgs)
            responses.append(resp)
            msgs.append(resp)
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
        draft = str(responses[-1].content) if responses else ""
        return {"draft": draft, "answer": draft,
                "responses": responses, "tool_names": tools_used}

    def _reflect(self, state: _State) -> dict:
        resp = self.llm.invoke(
            [SystemMessage(content=CRITIC_PROMPT),
             HumanMessage(content=f"Question: {state['question']}\n\nAnswer to review:\n{state['draft']}")]
        )
        return {"critique": str(resp.content).strip(),
                "reflections": state["reflections"] + 1, "responses": [resp]}

    def _route(self, state: _State) -> str:
        if state["critique"].upper().startswith("ACCEPT"):
            return "accept"
        if state["reflections"] >= self.max_reflections:
            return "accept"
        return "revise"

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        final = self.graph.invoke(
            {"question": question, "draft": "", "critique": "", "reflections": 0,
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
            trace=[f"reflections={final['reflections']}"],
        )
