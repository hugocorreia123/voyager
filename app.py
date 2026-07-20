"""Voyager — Streamlit app.

Tab 1: Evaluations dashboard (rendered from committed data/eval_summary.json —
       loads instantly, no API calls; works on Hugging Face Spaces).
Tab 2: Live demo — run any topology on your own question and see it answer with
       full instrumentation.

    uv run streamlit run app.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import voyager_friendly as vf

st.set_page_config(page_title="Voyager — Agent Topology Evaluation", layout="wide")

import voyager_theme as th
th.inject()
SUMMARY = Path("data/eval_summary.json")


@st.cache_data
def load_summary() -> dict:
    return json.loads(SUMMARY.read_text())


def render_dashboard() -> None:
    if not SUMMARY.exists():
        st.warning("No eval summary found. Run `uv run python -m voyager.eval.export`.")
        return
    s = load_summary()
    topo = s["topologies"]
    rows = sorted(topo.items(), key=lambda kv: -kv[1]["score"])

    vf.show_how_it_works()
    jv = s.get("judge_validation")
    if jv:
        vf.show_metrics(jv)

    st.subheader("Per-topology results")
    st.dataframe(
        [{"topology": t, "score": v["score"], "mean tokens": v["tokens"],
          "mean latency (s)": v["latency"],
          "Pareto": "★" if t in s["pareto"] else "", "n": v["n"]}
         for t, v in rows],
        use_container_width=True, hide_index=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Quality vs cost")
        fig = go.Figure()
        for t, v in topo.items():
            on = t in s["pareto"]
            fig.add_trace(go.Scatter(
                x=[v["tokens"]], y=[v["score"]], mode="markers+text",
                text=[t], textposition="top center",
                marker=dict(size=18 if on else 12,
                            symbol="star" if on else "circle"), name=t))
        fig.update_layout(xaxis_title="mean tokens (cost) →",
                          yaxis_title="mean score (quality) →", showlegend=False,
                          height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top-left is best. ★ = Pareto-optimal (nothing is both better and cheaper).")
    with col_r:
        st.subheader("Score by topology × category")
        cats = s["categories"]
        cs = s["category_scores"]
        ylabels = [t for t, _ in rows]
        z = [[cs[t].get(c) for c in cats] for t in ylabels]
        text = [["" if v is None else f"{v:.2f}" for v in row] for row in z]
        fig2 = go.Figure(data=go.Heatmap(
            z=z, x=cats, y=ylabels, zmin=0, zmax=1, colorscale="RdYlGn",
            text=text, texttemplate="%{text}", showscale=False))
        fig2.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Differentiation appears only in the post-cutoff 'recent' column.")


def render_live() -> None:
    from voyager.topologies import TOPOLOGIES

    st.subheader("Run an agent live")
    st.caption(
        "The evaluation artifacts were measured on qwen/qwen3-32b, which "
        "Groq retired in June 2026 — live runs now use its recommended "
        "replacement, openai/gpt-oss-120b. The Evaluations tab is "
        "unchanged: those numbers describe the measured run."
    )
    q = st.text_input(
        "Question",
        "What is the most recent Llama model and its benchmark results?",
    )
    names = list(TOPOLOGIES)
    t = st.selectbox("Topology", names,
                     index=names.index("react") if "react" in names else 0)

    if st.button("Run", type="primary"):
        result, last_err = None, None
        for attempt in range(3):
            try:
                with st.spinner(f"Running {t} … (plan-execute and "
                                "multi-agent can take a while)"):
                    result = TOPOLOGIES[t]().run(q)
                break
            except Exception as e:
                msg = str(e)
                if "tool_use_failed" in msg or "tool call validation" in msg.lower():
                    last_err = e
                    if attempt < 2:
                        st.caption(f"The model emitted a malformed tool "
                                   f"call — retrying ({attempt + 1}/2)…")
                    continue
                st.error("The live run failed before finishing.")
                with st.expander("Technical detail"):
                    st.write(f"{type(e).__name__}: {e}")
                st.info("Live runs need GROQ_API_KEY (a Space secret, "
                        "or .env locally).")
                return
        if result is None:
            st.error("The model produced a malformed tool call three "
                     "times in a row — press Run again, or try another "
                     "topology. This is a quirk of the replacement "
                     "model's tool calling, not of the agent design.")
            with st.expander("Technical detail"):
                st.write(f"{type(last_err).__name__}: {last_err}")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", result.steps)
        c2.metric("Tokens", result.total_tokens)
        c3.metric("Latency", f"{result.latency_s:.1f} s")
        c4.metric("Tool calls", len(result.tool_calls))
        st.caption("**Means:** what the answer cost. Fewer steps and "
                   "tokens at the same quality is the whole game — that "
                   "trade-off is exactly what the Evaluations tab measures.")

        st.markdown("#### Answer")
        st.markdown(result.answer)
        if result.tool_calls:
            st.caption("tools used: " + ", ".join(result.tool_calls))
        with st.expander("trace"):
            st.write(result.trace)


if not vf.show_welcome():
    st.stop()

th.hero(
    "Agent Topology Evaluation",
    "Voyager",
    "Five agent designs, one shared model, one judge validated against "
    "blind human labels — quality weighed against cost, not admired "
    "in isolation.",
    "qwen3-32b agents · gpt-oss-120b judge · κ = 0.95 vs humans",
)

tab1, tab2, tab3 = st.tabs(["Evaluations", "Live demo", "❓ Help"])
with tab1:
    render_dashboard()
with tab2:
    render_live()
with tab3:
    vf.show_help()
