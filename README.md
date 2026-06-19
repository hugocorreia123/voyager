# Voyager

**An agentic-research platform that benchmarks five LLM agent topologies against an evaluation whose LLM judge is itself validated against human labels — Cohen's κ = 0.95.**

Building one agent is a weekend. The hard, rare part is *measuring* agents: designing an evaluation you can actually trust, and then using it to say which agent design wins, on which kind of task, and at what cost. That evaluation — not the agents — is the point of this project.

---

## Headline findings

Five topologies, one shared model (`qwen/qwen3-32b` via Groq) and one shared toolset, run over a research benchmark and graded by a **separate** judge model (`openai/gpt-oss-120b`) whose grades were validated against hand labels.

| topology | mean score | mean tokens | mean latency |
|---|---:|---:|---:|
| **react** | **0.95** | **2,244** | **31.0 s** |
| multi_agent | 0.95 | 3,809 | 34.1 s |
| plan_execute | 0.95 | 7,364 | 69.4 s |
| self_reflective | 0.55 | 3,196 | 33.3 s |
| single_shot (baseline) | 0.50 | 843 | 2.5 s |

**1. The simplest agent wins.** ReAct matches the elaborate topologies on quality (0.95) at a third of plan-execute's token cost and the lowest latency of any tool user. On a quality-vs-cost Pareto frontier, only **ReAct** and the no-tool baseline survive; multi-agent, plan-execute, and self-reflective are all *dominated* — something is both better **and** cheaper than each of them.

**2. Self-reflection actively hurt.** It scored **0.55** while costing *more* tokens than ReAct. On the questions that require fresh information it collapsed to **0.10** (vs ReAct's 0.90): the agent searched, found the correct answer, and then its own critic — sharing the base model's stale prior — talked it back out of the answer and into a confident "that doesn't exist" hallucination. Reflection spent more compute to produce worse answers.

**3. A benchmark only measures topology where the model is ignorant.** Broken out by category, *every* topology scores a perfect 1.00 on the five categories whose answers sit in the model's training data. All differentiation lives in the `recent` category (post-training-cutoff facts):

| topology | in-training categories (×5) | `recent` (post-cutoff) |
|---|---:|---:|
| react / plan_execute / multi_agent | 1.00 | 0.90 |
| self_reflective | 1.00 | 0.10 |
| single_shot | 1.00 | 0.00 |

When the answer is already in parametric memory, tools and topology are irrelevant. Agent design only matters when the model has to *find out*.

---

## Why you can trust these numbers

The findings above are only worth as much as the judge that produced them. So the judge was validated, not assumed:

- The judge is a **different model family** from the agents (`gpt-oss` grading `qwen`), to avoid a model rewarding its own style.
- Every answer was **hand-labelled by a human**, blind to the judge's verdict.
- Judge vs human over 50 pairs: **Cohen's κ = 0.95 ("almost perfect"), 98 % raw agreement, one disagreement.**

```
Confusion matrix (rows = human, cols = judge)
                 CORR     PART     INCO
CORRECT            37        1        0
PARTIAL             0        3        0
INCORRECT           0        0        9
```

That single number — κ = 0.95 — is what separates this from an agent demo. Run `agreement` yourself (below) to reproduce it.

---

## The five topologies

All share the same model and tools, so measured differences are attributable to *topology*, not to a different LLM or toolset.

- **single_shot** — answers directly, no tools. The baseline.
- **react** — reason → act (tool) → observe, looped.
- **plan_execute** — plan the steps upfront, execute each with tools, then synthesize.
- **self_reflective** — draft, then a critic reviews and triggers revision (capped).
- **multi_agent** — an orchestrator routes sub-questions to role-specialized agents (web / academic), then an aggregator synthesizes.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a free [Groq API key](https://console.groq.com/keys).

```bash
cp .env.example .env          # paste your GROQ_API_KEY
uv sync
```

## Run a single question

```bash
uv run voyager "What is the most recent Llama model and its benchmark results?"
uv run voyager -t self_reflective "..."     # pick a topology
```

Output is fully instrumented:

```
--- Answer [react] ---
<grounded answer>
steps=2  tools=['web_search']  tokens(in/out)=1121/1074  latency=5.5s
```

## Reproduce the evaluation

```bash
uv run python -m voyager.eval.runner        # run all 5 topologies over the benchmark (resumable)
uv run python -m voyager.eval.judge         # LLM-judge each answer against its rubric
uv run python -m voyager.eval.label         # hand-label answers yourself (judge-blind)
uv run python -m voyager.eval.agreement     # judge vs human: Cohen's kappa
uv run python -m voyager.eval.analysis      # per-topology scores, cost, Pareto frontier
```

---

## How it's built

```
voyager/
├── config.py                 # shared LLM (one model across all topologies — a methodological rule)
├── tools.py                  # shared tools: web_search (DuckDuckGo), arxiv_search
├── cli.py                    # run one question through a chosen topology
├── benchmark.py              # load the benchmark (stable, verifiable ground truth)
├── topologies/
│   ├── base.py               # Topology ABC + RunResult (the contract the eval consumes)
│   ├── single_shot.py  react.py  plan_execute.py  self_reflective.py  multi_agent.py
└── eval/
    ├── runner.py             # topologies × benchmark → results.jsonl (resumable, rate-limit tolerant)
    ├── judge.py              # cross-family LLM judge → judgments.jsonl
    ├── label.py              # human labelling tool (judge-blind)
    ├── agreement.py          # Cohen's kappa, confusion matrix
    └── analysis.py           # per-topology quality/cost, Pareto frontier
```

**Stack:** Python · uv · LangGraph · Groq (`qwen/qwen3-32b` agents, `openai/gpt-oss-120b` judge) · DuckDuckGo + arXiv tools.

**Benchmark design.** Questions have **stable, verifiable ground truth** (no "latest/most recent" phrasings whose answer drifts), split into in-training control facts and post-cutoff facts that *require* retrieval — the latter being the only ones that discriminate between topologies.

---

## Scope & honesty

This is a research-quality framework demonstrated on a **10-question seed benchmark (5 control + 5 research-required), single seed per run.** The results are **descriptive, not yet statistically significant** — the point so far is a *trustworthy* pipeline (validated judge, clean cost/quality accounting, reproducible), not a large-sample claim. The framework scales directly to more questions and multiple seeds; significance testing and confidence intervals are the natural next step, along with a Streamlit app for the live agent-graph view and an evals dashboard.
