"""ReAct topology: reason -> act (tool) -> observe, looped until the model
answers without requesting a tool.

Built as an explicit LangGraph graph (not langgraph.prebuilt) so every step is
instrumented for the eval harness. The other four topologies will follow this
same shape: build a graph, expose .run() -> RunResult.
"""
from __future__ import annotations

import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from voyager.config import get_llm
from voyager.tools import TOOLS
from voyager.topologies.base import RunResult, Topology

SYSTEM_PROMPT = (
    "You are a rigorous research assistant. Use the available tools to gather "
    "evidence before answering. Search when you are unsure or when the question "
    "concerns recent events. When you have enough information, give a clear, "
    "well-supported final answer and cite what you found."
)

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class _State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReActTopology(Topology):
    name = "react"

    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self.llm = get_llm().bind_tools(TOOLS)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(_State)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_edge(START, "agent")
        g.add_conditional_edges(
            "agent", self._route, {"tools": "tools", "end": END}
        )
        g.add_edge("tools", "agent")
        return g.compile()

    # --- nodes -------------------------------------------------------------
    def _agent(self, state: _State) -> dict:
        return {"messages": [self.llm.invoke(state["messages"])]}

    def _tools(self, state: _State) -> dict:
        last = state["messages"][-1]
        out: list[AnyMessage] = []
        for call in last.tool_calls:
            tool = _TOOLS_BY_NAME[call["name"]]
            try:
                result = tool.invoke(call["args"])
            except Exception as e:  # tools must never crash the loop
                result = f"Tool error: {e}"
            out.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
        return {"messages": out}

    def _route(self, state: _State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    # --- public API --------------------------------------------------------
    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        init = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        }
        final = self.graph.invoke(
            init, {"recursion_limit": self.max_steps * 2}
        )
        latency = time.perf_counter() - t0

        msgs = final["messages"]
        in_tok = out_tok = steps = 0
        tool_calls: list[str] = []
        for m in msgs:
            usage = getattr(m, "usage_metadata", None)
            if usage:
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
                steps += 1
            for tc in getattr(m, "tool_calls", []) or []:
                tool_calls.append(tc["name"])

        return RunResult(
            topology=self.name,
            question=question,
            answer=str(msgs[-1].content),
            steps=steps,
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            trace=[type(m).__name__ for m in msgs],
        )
