"""Export a compact eval summary JSON for the Streamlit dashboard (and HF Spaces),
so the app renders instantly from committed data without re-running anything.

    uv run python -m voyager.eval.export   ->  data/eval_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from voyager.eval.agreement import ORDER, cohen_kappa
from voyager.eval.analysis import _load_judgments
from voyager.eval.judge import RESULTS, load_results
from voyager.eval.label import LABELS

OUT = Path("data/eval_summary.json")


def build_summary() -> dict:
    results = {(r["topology"], r["question_id"], r["seed"]): r
               for r in load_results(RESULTS)}
    judg = _load_judgments()
    keys = sorted(set(results) & set(judg))

    by_topo = defaultdict(lambda: {"score": [], "tokens": [], "latency": []})
    by_tc = defaultdict(list)
    for k in keys:
        r, jd = results[k], judg[k]
        t = r["topology"]
        by_topo[t]["score"].append(jd["score"])
        by_topo[t]["tokens"].append(r.get("total_tokens", 0))
        by_topo[t]["latency"].append(r.get("latency_s", 0.0))
        by_tc[(t, r["category"])].append(jd["score"])

    topo = {t: {"score": round(mean(d["score"]), 3),
                "tokens": round(mean(d["tokens"])),
                "latency": round(mean(d["latency"]), 1),
                "n": len(d["score"])}
            for t, d in by_topo.items()}

    pts = {t: (v["score"], v["tokens"]) for t, v in topo.items()}
    frontier = [t for t, (s, tok) in pts.items()
                if not any((s2 >= s and tok2 <= tok and (s2 > s or tok2 < tok))
                           for t2, (s2, tok2) in pts.items() if t2 != t)]

    cats = sorted({c for (_, c) in by_tc})
    category_scores = {
        t: {c: (round(mean(by_tc[(t, c)]), 3) if by_tc.get((t, c)) else None)
            for c in cats}
        for t in topo
    }

    summary = {"topologies": topo, "pareto": frontier,
               "categories": cats, "category_scores": category_scores}

    if LABELS.exists():
        human = {}
        for line in open(LABELS, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                human[(d["topology"], d["question_id"], d["seed"])] = d["human_verdict"]
        jverd = {k: judg[k]["verdict"] for k in judg}
        pk = sorted(set(human) & set(jverd))
        if pk:
            h = [human[k] for k in pk]
            j = [jverd[k] for k in pk]
            po, kappa = cohen_kappa(h, j, ORDER)
            cm = [[sum(1 for hh, jj in zip(h, j) if hh == hl and jj == jl)
                   for jl in ORDER] for hl in ORDER]
            summary["judge_validation"] = {"n": len(pk), "agreement": round(po, 3),
                                           "kappa": round(kappa, 3),
                                           "labels": ORDER, "confusion": cm}
    return summary


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = build_summary()
    OUT.write_text(json.dumps(s, indent=2))
    print(f"wrote {OUT}")
    jv = s.get("judge_validation")
    if jv:
        print(f"judge: kappa={jv['kappa']}  agreement={jv['agreement']}  n={jv['n']}")


if __name__ == "__main__":
    main()
