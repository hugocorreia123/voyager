"""Plan-and-execute topology.

Plan the research upfront, execute each step (with tools), then synthesize a
final answer. Differs from ReAct by separating planning and synthesis from the
acting loop — which tends to help on multi-hop questions and hurt on simple ones.
"""
from __future__ import annotations

import operator
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from voyager.config import get_llm
from voyager.tools import TOOLS
from voyager.topologies.base import RunResult, Topology

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

PLANNER_PROMPT = (
    "You are a research planner. Break the user's question into 2-4 concise, "
    "ordered research steps that, once executed, will let you answer it. "
    "Return ONLY the steps, one per line, with no numbering or extra text."
)
EXECUTOR_PROMPT = (
    "You are a research executor. Carry out the given step, using the tools when "
    "needed (search the web for current facts). Return a short factual result for "
    "this step only."
)
SYNTH_PROMPT = (
    "You are a research assistant. Using the findings from each step, write a "
    "clear, well-supported final answer to the original question."
)


class _State(TypedDict):
    question: str
    plan: list[str]
    step_idx: int
    step_results: Annotated[list[str], operator.add]
    answer: str
    responses: Annotated[list, operator.add]
    tool_names: Annotated[list[str], operator.add]


class PlanExecuteTopology(Topology):
    name = "plan_execute"

    def __init__(self, executor_max_rounds: int = 3) -> None:
        self.executor_max_rounds = executor_max_rounds
        self.llm = get_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(_State)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("synthesize", self._synthesize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "execute")
        g.add_conditional_edges(
            "execute", self._more,
            {"execute": "execute", "synthesize": "synthesize"},
        )
        g.add_edge("synthesize", END)
        return g.compile()

    def _plan(self, state: _State) -> dict:
        resp = self.llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT),
             HumanMessage(content=state["question"])]
        )
        steps = []
        for line in str(resp.content).splitlines():
            s = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip()
            if s:
                steps.append(s)
        steps = steps[:4] or [state["question"]]
        return {"plan": steps, "step_idx": 0, "responses": [resp]}

    def _execute(self, state: _State) -> dict:
        step = state["plan"][state["step_idx"]]
        msgs = [
            SystemMessage(content=EXECUTOR_PROMPT),
            HumanMessage(
                content=f"Original question: {state['question']}\n\nStep: {step}"
            ),
        ]
        responses, tools_used = [], []
        for _ in range(self.executor_max_rounds):
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
        result_text = str(responses[-1].content) if responses else ""
        return {
            "step_results": [f"{step}: {result_text}"],
            "step_idx": state["step_idx"] + 1,
            "responses": responses,
            "tool_names": tools_used,
        }

    def _more(self, state: _State) -> str:
        return "execute" if state["step_idx"] < len(state["plan"]) else "synthesize"

    def _synthesize(self, state: _State) -> dict:
        findings = "\n\n".join(state["step_results"])
        resp = self.llm.invoke(
            [SystemMessage(content=SYNTH_PROMPT),
             HumanMessage(content=f"Question: {state['question']}\n\nFindings:\n{findings}")]
        )
        return {"answer": str(resp.content), "responses": [resp]}

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        final = self.graph.invoke(
            {"question": question, "plan": [], "step_idx": 0,
             "step_results": [], "answer": "", "responses": [], "tool_names": []},
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
            trace=final["plan"],
        )
