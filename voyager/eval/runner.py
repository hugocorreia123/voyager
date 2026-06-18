"""Eval runner: execute each topology over the benchmark and persist raw
results (answers + instrumentation). No scoring here — the judge (next step)
reads these results and scores them.

Resumable: re-running skips (topology, question_id, seed) combos already present
in the output file, so you can Ctrl-C and continue (useful with Groq rate limits).

    uv run python -m voyager.eval.runner                        # all topologies, 1 seed
    uv run python -m voyager.eval.runner -t single_shot react   # subset
    uv run python -m voyager.eval.runner --seeds 3              # 3 runs per question
    uv run python -m voyager.eval.runner -t react --limit 2     # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from voyager.benchmark import load_benchmark
from voyager.topologies import TOPOLOGIES

OUT = Path("runs/results.jsonl")


def _existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d.get("error") is None:  # only successful runs count as done
                        keys.add((d["topology"], d["question_id"], d["seed"]))
    return keys


def run_benchmark(topo_names, seeds, limit, out: Path = OUT) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    questions = load_benchmark()
    if limit:
        questions = questions[:limit]
    done = _existing_keys(out)
    total = len(topo_names) * len(questions) * seeds
    n = 0
    for tname in topo_names:
        topo = TOPOLOGIES[tname]()  # build once per topology
        for seed in range(seeds):
            for q in questions:
                n += 1
                if (tname, q.id, seed) in done:
                    print(f"[{n}/{total}] skip {tname}/{q.id}/seed{seed}")
                    continue
                print(f"[{n}/{total}] run  {tname}/{q.id}/seed{seed} ...",
                      end=" ", flush=True)
                rec = {
                    "topology": tname, "question_id": q.id, "category": q.category,
                    "seed": seed, "question": q.question,
                    "reference_answer": q.reference_answer, "rubric": q.rubric,
                }
                try:
                    r = topo.run(q.question)
                    rec.update(
                        answer=r.answer, steps=r.steps, tool_calls=r.tool_calls,
                        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                        total_tokens=r.total_tokens, latency_s=round(r.latency_s, 2),
                        error=None,
                    )
                    print(f"ok ({rec['latency_s']}s, {rec['total_tokens']} tok)")
                except Exception as e:
                    rec.update(
                        answer=None, steps=0, tool_calls=[], input_tokens=0,
                        output_tokens=0, total_tokens=0, latency_s=0.0,
                        error=f"{type(e).__name__}: {e}",
                    )
                    print(f"ERROR {type(e).__name__}: {e}")
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nDone. Results -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="voyager-eval")
    p.add_argument("-t", "--topologies", nargs="*", default=list(TOPOLOGIES),
                   choices=list(TOPOLOGIES), help="topologies to run (default: all)")
    p.add_argument("--seeds", type=int, default=1,
                   help="runs per question per topology")
    p.add_argument("--limit", type=int, default=0,
                   help="limit number of questions (0 = all)")
    args = p.parse_args()
    run_benchmark(args.topologies, args.seeds, args.limit or None)


if __name__ == "__main__":
    main()
