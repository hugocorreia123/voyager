"""Judge validation: how well does the LLM judge agree with the human labels?

Computes raw agreement and Cohen's kappa over the (topology, question_id, seed)
pairs labeled by both human and judge. kappa >= ~0.6-0.7 means the judge is
trustworthy enough to use at scale; below that, fix the rubrics or judge prompt.

    uv run python -m voyager.eval.agreement
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from voyager.eval.judge import JUDGMENTS
from voyager.eval.label import LABELS

ORDER = ["CORRECT", "PARTIAL", "INCORRECT"]


def _load(path: Path, key: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[(d["topology"], d["question_id"], d["seed"])] = d[key]
    return out


def cohen_kappa(human: list, judge: list, labels: list) -> tuple[float, float]:
    n = len(human)
    po = sum(1 for h, j in zip(human, judge) if h == j) / n
    ch, cj = Counter(human), Counter(judge)
    pe = sum((ch[l] / n) * (cj[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return po, kappa


def interpret(k: float) -> str:
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> None:
    judge = _load(JUDGMENTS, "verdict")
    human = _load(LABELS, "human_verdict")
    keys = sorted(set(judge) & set(human))
    if not keys:
        print("No overlapping labeled pairs. Run the judge and the labeler first.")
        return
    h = [human[k] for k in keys]
    j = [judge[k] for k in keys]
    po, kappa = cohen_kappa(h, j, ORDER)

    print(f"Pairs compared: {len(keys)}")
    print(f"Raw agreement:  {po:.1%}")
    print(f"Cohen's kappa:  {kappa:.3f}  ({interpret(kappa)})")

    print("\nConfusion matrix (rows = human, cols = judge):")
    print(f"{'':12s}" + "".join(f"{l[:4]:>9s}" for l in ORDER))
    for hl in ORDER:
        row = [sum(1 for hh, jj in zip(h, j) if hh == hl and jj == jl)
               for jl in ORDER]
        print(f"{hl:12s}" + "".join(f"{c:9d}" for c in row))

    diffs = [(k, human[k], judge[k]) for k in keys if human[k] != judge[k]]
    if diffs:
        print(f"\nDisagreements ({len(diffs)}):")
        for (t, q, s), hh, jj in diffs:
            print(f"  {t}/{q}: human={hh}  judge={jj}")

    ok = kappa >= 0.6
    print(f"\nJudge is {'TRUSTWORTHY' if ok else 'NOT trustworthy — revise rubrics/judge'} "
          f"(threshold kappa >= 0.6).")


if __name__ == "__main__":
    main()
