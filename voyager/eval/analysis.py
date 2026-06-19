"""Analysis: turn raw runs + validated judgments into the headline results —
per-topology quality, cost (tokens/latency), the quality-vs-cost Pareto frontier,
and a topology x category breakdown.

    uv run python -m voyager.eval.analysis
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from voyager.eval.judge import JUDGMENTS, RESULTS, load_results


def _load_judgments(path: Path = JUDGMENTS) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[(d["topology"], d["question_id"], d["seed"])] = d
    return out


def main() -> None:
    results = {(r["topology"], r["question_id"], r["seed"]): r
               for r in load_results(RESULTS)}
    judg = _load_judgments()
    keys = sorted(set(results) & set(judg))
    if not keys:
        print("No matched runs+judgments. Run the runner and judge first.")
        return

    by_topo = defaultdict(lambda: {"score": [], "tokens": [], "latency": []})
    by_tc = defaultdict(list)
    for k in keys:
        r, jd = results[k], judg[k]
        t = r["topology"]
        by_topo[t]["score"].append(jd["score"])
        by_topo[t]["tokens"].append(r.get("total_tokens", 0))
        by_topo[t]["latency"].append(r.get("latency_s", 0.0))
        by_tc[(t, r["category"])].append(jd["score"])

    rows = sorted(
        ((t, mean(d["score"]), mean(d["tokens"]), mean(d["latency"]), len(d["score"]))
         for t, d in by_topo.items()),
        key=lambda x: -x[1],
    )

    print("=== Per-topology summary ===")
    print(f"{'topology':16s}{'score':>7s}{'tokens':>9s}{'latency':>9s}{'n':>4s}")
    for t, s, tok, lat, n in rows:
        print(f"{t:16s}{s:7.2f}{tok:9.0f}{lat:8.1f}s{n:4d}")

    pts = {t: (s, tok) for t, s, tok, lat, n in rows}
    frontier = [
        t for t, (s, tok) in pts.items()
        if not any((s2 >= s and tok2 <= tok and (s2 > s or tok2 < tok))
                   for t2, (s2, tok2) in pts.items() if t2 != t)
    ]
    print("\nPareto-optimal (max quality, min token cost): "
          + ", ".join(sorted(frontier, key=lambda t: -pts[t][0])))
    dominated = [t for t in pts if t not in frontier]
    if dominated:
        print("Dominated (something is both better AND cheaper): "
              + ", ".join(dominated))

    cats = sorted({c for (_, c) in by_tc})
    print("\n=== Mean score by topology x category ===")
    print(f"{'topology':16s}" + "".join(f"{c[:7]:>9s}" for c in cats))
    for t, *_ in rows:
        print(f"{t:16s}" + "".join(
            f"{mean(by_tc[(t, c)]):9.2f}" if by_tc.get((t, c)) else f"{'-':>9s}"
            for c in cats))


if __name__ == "__main__":
    main()
