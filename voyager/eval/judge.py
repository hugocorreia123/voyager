"""LLM-as-judge: score each agent answer against the rubric + reference answer.

Design choice: the judge uses a DIFFERENT model family from the agents
(gpt-oss judging qwen). A model grading its own family tends to score itself up;
a different judge reduces that self-preference bias. The judge calls no tools, so
gpt-oss's browser-tool quirk is irrelevant here.

This judge is NOT trusted until Phase 4 validates it against human labels.

    uv run python -m voyager.eval.judge
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from voyager.config import get_llm

JUDGE_MODEL = "openai/gpt-oss-120b"
RESULTS = Path("runs/results.jsonl")
JUDGMENTS = Path("runs/judgments.jsonl")

JUDGE_PROMPT = (
    "You are a strict, fair grader. You are given a question, a reference answer, "
    "a rubric, and a candidate answer. Decide how well the candidate answer "
    "satisfies the rubric relative to the reference answer. Judge factual "
    "correctness only — ignore style, length, and extra detail that does not "
    "contradict the reference. Respond with EXACTLY one of these words on the "
    "first line: CORRECT, PARTIAL, or INCORRECT. On the second line, give a "
    "one-sentence justification."
)

_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def load_results(path: Path = RESULTS) -> list[dict]:
    """Dedupe by (topology, question_id, seed): prefer successful, last wins."""
    by_key: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["topology"], d["question_id"], d["seed"])
            if key not in by_key or d.get("error") is None:
                by_key[key] = d
    return list(by_key.values())


def _parse_verdict(text: str) -> tuple[str, float]:
    first = (text.splitlines()[0].strip().upper() if text.strip() else "")
    if "INCORRECT" in first:
        v = "INCORRECT"
    elif "PARTIAL" in first:
        v = "PARTIAL"
    elif "CORRECT" in first:
        v = "CORRECT"
    else:
        v = "INCORRECT"
    return v, _SCORE[v]


def judge_one(judge, r: dict) -> tuple[str, float, str]:
    if r.get("error") or not r.get("answer"):
        return "INCORRECT", 0.0, "No answer produced (run errored)."
    msg = (
        f"Question:\n{r['question']}\n\n"
        f"Reference answer:\n{r['reference_answer']}\n\n"
        f"Rubric:\n{r['rubric']}\n\n"
        f"Candidate answer:\n{r['answer']}"
    )
    resp = judge.invoke(
        [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=msg)]
    )
    text = str(resp.content).strip()
    verdict, score = _parse_verdict(text)
    lines = text.splitlines()
    justification = (" ".join(lines[1:]).strip() or text)[:300]
    return verdict, score, justification


def _judged_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    keys.add((d["topology"], d["question_id"], d["seed"]))
    return keys


def judge_all(results_path: Path = RESULTS, out: Path = JUDGMENTS) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    judge = get_llm(model=JUDGE_MODEL)
    results = load_results(results_path)
    done = _judged_keys(out)
    total = len(results)
    for i, r in enumerate(results, 1):
        key = (r["topology"], r["question_id"], r["seed"])
        if key in done:
            print(f"[{i}/{total}] skip {r['topology']}/{r['question_id']}")
            continue
        print(f"[{i}/{total}] judge {r['topology']}/{r['question_id']} ...",
              end=" ", flush=True)
        verdict, score, why = judge_one(judge, r)
        print(f"{verdict} ({score})")
        rec = {
            "topology": r["topology"], "question_id": r["question_id"],
            "category": r["category"], "seed": r["seed"],
            "verdict": verdict, "score": score, "justification": why,
        }
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _summary(out)


def _summary(path: Path) -> None:
    scores: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                scores[d["topology"]].append(d["score"])
    print("\n=== mean score by topology (judge — NOT yet human-validated) ===")
    for t, s in sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {t:16s} {sum(s)/len(s):.2f}  (n={len(s)})")


def main() -> None:
    argparse.ArgumentParser(prog="voyager-judge").parse_args()
    judge_all()


if __name__ == "__main__":
    main()
