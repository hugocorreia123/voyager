"""Voyager — friendly UI layer.

Welcome tour, plain-language metric explanations, honesty box, and a help
tab. Streamlit only — no other dependencies. Same pattern as Mandate's
friendly UI, in Voyager's voice: written for someone choosing an agent
architecture.
"""

import streamlit as st

_TOUR_KEY = "voyager_tour_done"


# ------------------------------------------------------------------ welcome
def show_welcome() -> bool:
    """One-time plain-language tour. Returns True once dismissed."""
    if st.session_state.get(_TOUR_KEY):
        return True

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.title("🧭 Voyager")
        st.subheader("Before trusting a judge to rank your agents, check the judge.")
        st.markdown(
            """
Everyone can build an AI agent. The hard, rare part is **measuring**
agents: an evaluation you can actually trust, that says which agent
design wins, on which kind of task, and at what cost. Voyager benchmarks
**five agent topologies** on one shared model — and, before believing any
score, validates the AI judge doing the grading **against blind human
labels**.

**What you can do here**

- **Evaluations** — the full results: is the judge trustworthy, which
  topology wins, and the quality-vs-cost trade-off (the ★ marks are the
  Pareto frontier: nothing is both better and cheaper).
- **Live demo** — ask your own question, pick a topology, and watch it
  answer with full instrumentation (steps, tokens, latency, tools).
- **❓ Help** — every term on screen, in plain language.

**What this will not tell you**

Which topology is "best" in general. The honest answer is *it depends on
the task* — the results show exactly where the designs differ and where
they don't.
"""
        )
        c1, c2 = st.columns(2)
        if c1.button("Start exploring", type="primary", use_container_width=True):
            st.session_state[_TOUR_KEY] = True
            st.rerun()
        if c2.button("Skip the intro", use_container_width=True):
            st.session_state[_TOUR_KEY] = True
            st.rerun()
    return False


def show_replay() -> None:
    """Small control to replay the intro tour."""
    if st.button("↻ Replay the intro", help="Show the welcome tour again"):
        st.session_state[_TOUR_KEY] = False
        st.rerun()


# ------------------------------------------------------------------ metrics
def show_metrics(jv: dict) -> None:
    """The judge-trust header with a plain-language 'means' line per metric.

    Takes the live judge_validation dict — values are never hardcoded.
    """
    st.subheader("Is the judge trustworthy?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Judge ↔ human (Cohen's κ)", jv["kappa"])
        st.caption(
            "**Means:** the AI judge and a human agree far beyond what "
            "chance would produce — near the top of the scale, so the "
            "rankings below rest on solid ground."
        )
    with c2:
        st.metric("Raw agreement", f"{jv['agreement'] * 100:.0f}%")
        st.caption(
            "**Means:** on the blind-labelled answers, judge and human "
            f"gave the same verdict {jv['agreement'] * 100:.0f}% of the time."
        )
    with c3:
        st.metric("Human-labelled pairs", jv["n"])
        st.caption(
            f"**Means:** a human graded {jv['n']} answers without seeing "
            "the judge's scores — that blindness is what makes the "
            "comparison honest."
        )
    st.caption(
        "The judge is a different model family from the agents (no "
        "self-preference) and was validated before any topology was ranked. "
        "Everything below depends on it."
    )


# ------------------------------------------------------------------ method
def show_how_it_works() -> None:
    """Plain-language story for the top of the Evaluations tab."""
    with st.expander("How this evaluation works, in one breath"):
        st.markdown(
            """
Five agent designs — from a single direct call to multi-agent
orchestration — answer the **same benchmark questions with the same
underlying model**, so the only variable is the *topology*. An LLM judge
from a **different model family** grades every answer; the judge itself
was first validated against **blind human labels** (the κ above). Each
run is fully instrumented — steps, tokens, latency, tool calls — so
quality can be weighed against cost, not admired in isolation.

**Does this affect what you see?** Yes, in one specific way: scores only
mean something because the judge was checked first. An unvalidated judge
can rank confidently and wrongly — checking the instrument before
believing the measurement is the whole method here.
"""
        )


# ------------------------------------------------------------------ help
def show_help(summary: dict | None = None) -> None:
    """Help tab: orientation, mini-glossary, and the honesty box."""
    st.header("❓ Help")

    st.markdown(
        """
#### What am I looking at?

A benchmark of five AI-agent architectures ("topologies") on identical
questions with an identical model, graded by a human-validated AI judge.

**Start here:** open **Evaluations** and read the judge-trust row first —
then the per-topology table, then the quality-vs-cost chart.

#### Words on the screen, in plain language

- **Topology** — the agent's architecture: how it plans, calls tools,
  and loops (e.g. a single call vs ReAct vs plan-and-execute vs
  multi-agent).
- **Cohen's κ (kappa)** — agreement between judge and human, corrected
  for chance. 1.0 is perfect; 0 is coin-flipping.
- **Blind labels** — the human graded answers without seeing the
  judge's scores.
- **Pareto frontier (★)** — the designs where nothing else is both
  better *and* cheaper. If it's not on the frontier, something dominates
  it.
- **Tokens** — the cost currency: what you pay and wait for.
- **Cross-family judge** — the grader is a different model family from
  the agents, so it cannot favour its own style.
"""
    )

    st.markdown("#### The honesty box")
    st.markdown(
        """
- **Topologies differentiate less than the hype suggests.** On most
  question categories the designs score similarly — real differences
  appear mainly on post-cutoff "recent" questions, where tool use
  matters. The heatmap shows this rather than hiding it.
- **The judge is validated, not infallible.** κ was measured on this
  benchmark's label set; a different task family would need its own
  validation round.
- **The live demo needs an API key** (GROQ_API_KEY). Without it, the
  committed evaluation results still load instantly — nothing on the
  Evaluations tab depends on a key.
- **Cost numbers are means over the benchmark**, not guarantees for
  your workload.
"""
    )

    st.markdown(
        """
#### Links

[GitHub repository](https://github.com/hugocorreia123/voyager)
· Hugo Correia — [LinkedIn](https://www.linkedin.com/in/hugogncorreia)
"""
    )
    show_replay()
