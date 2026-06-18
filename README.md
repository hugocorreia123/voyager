# Voyager

Agentic research platform that implements **five agent topologies** and rigorously
compares them — with an LLM judge **validated against human labels** and
**statistical** significance testing — on a 200-question research benchmark.

> The agent is the easy part. The *evaluation* is the project. See `Voyager_Build_Spec.md`.

## Status

**Phase 0 — scaffold (this commit):** shared model + tool layer, and a working
**ReAct** topology answering a single question end-to-end with full instrumentation
(steps, tool calls, tokens, latency).

Next: the other four topologies (single-shot, plan-and-execute, multi-agent,
self-reflective), then the benchmark, eval harness, judge validation, stats, and
the Streamlit app.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a free Groq key
(https://console.groq.com/keys).

```bash
cp .env.example .env          # then paste your GROQ_API_KEY
uv sync                       # creates .venv and installs deps
```

## Run

```bash
uv run voyager "What is the most recent Llama model and its benchmark results?"
```

Example output:

```
Q: What is the most recent Llama model and its benchmark results?

--- Answer [react] ---
<grounded answer>

steps=3  tools=['web_search', 'web_search']  tokens(in/out)=4120/380  latency=6.2s
```

## Layout

```
voyager/
├── config.py              # shared LLM (same model across all topologies)
├── tools.py               # shared tools (web_search, arxiv_search)
├── cli.py                 # run one question through a topology
└── topologies/
    ├── base.py            # Topology ABC + RunResult (the eval contract)
    └── react.py           # ReAct: reason -> act -> observe loop
```

## Design rule

The **model and tools are shared** across every topology so that measured
differences are attributable to *topology*, not to a different LLM or toolset.
That constraint is what makes the Phase-5 comparison valid.
