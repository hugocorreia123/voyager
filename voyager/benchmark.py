"""Benchmark loading.

Each line of data/benchmark.jsonl is a question with a STABLE, verifiable
ground-truth answer and a scoring rubric. 'Stable' is the key word: avoid
'latest / most recent' phrasings whose correct answer drifts over time, or the
benchmark rots. Answering still requires research — the ground truth just does
not move.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "benchmark.jsonl"

CATEGORIES = ["multi_hop", "quantitative", "comparison", "temporal", "tool_use"]


@dataclass
class Question:
    id: str
    category: str
    question: str
    reference_answer: str
    rubric: str


def load_benchmark(path: Path = DATA) -> list[Question]:
    out: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Question(**json.loads(line)))
    return out


if __name__ == "__main__":
    qs = load_benchmark()
    print(f"{len(qs)} questions; by category: {dict(Counter(q.category for q in qs))}")
