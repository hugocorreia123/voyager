"""Human labeling tool for judge validation (Phase 4).

You label each (question, candidate answer) yourself as correct/partial/incorrect
WITHOUT seeing the LLM judge's verdict — so the comparison isn't anchored to the
judge. Then agreement.py computes how well the judge matches you (Cohen's kappa).

Resumable: skips pairs you have already labeled. Type q to stop and resume later.

    uv run python -m voyager.eval.label
"""
from __future__ import annotations

import json
from pathlib import Path

from voyager.eval.judge import RESULTS, load_results

LABELS = Path("runs/human_labels.jsonl")
_MAP = {"c": ("CORRECT", 1.0), "p": ("PARTIAL", 0.5), "i": ("INCORRECT", 0.0)}


def _labeled_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                keys.add((d["topology"], d["question_id"], d["seed"]))
    return keys


def main() -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(RESULTS)
    done = _labeled_keys(LABELS)
    todo = [r for r in results
            if (r["topology"], r["question_id"], r["seed"]) not in done]
    print(f"{len(done)} already labeled, {len(todo)} to go.\n")

    for i, r in enumerate(todo, 1):
        print("=" * 72)
        print(f"({i}/{len(todo)})  [{r['topology']} / {r['question_id']}]")
        print(f"\nQUESTION:\n{r['question']}")
        print(f"\nREFERENCE:\n{r['reference_answer']}")
        print(f"\nRUBRIC:\n{r['rubric']}")
        ans = r.get("answer") or "(no answer — run errored)"
        if len(ans) > 2000:
            ans = ans[:2000] + " …[truncated]"
        print(f"\nCANDIDATE ANSWER:\n{ans}")

        while True:
            choice = input(
                "\nYour label — [c]orrect / [p]artial / [i]ncorrect / [s]kip / [q]uit: "
            ).strip().lower()
            if choice in _MAP or choice in ("s", "q"):
                break
            print("  please type c, p, i, s, or q")

        if choice == "q":
            print("Stopped. Re-run to resume.")
            return
        if choice == "s":
            continue

        verdict, score = _MAP[choice]
        rec = {"topology": r["topology"], "question_id": r["question_id"],
               "seed": r["seed"], "human_verdict": verdict, "human_score": score}
        with open(LABELS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  saved: {verdict}")

    print("\nAll done labeling.")


if __name__ == "__main__":
    main()
