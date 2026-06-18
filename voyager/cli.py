"""Run a single question through a chosen topology.

    uv run voyager "your question"                  # default: react
    uv run voyager -t single_shot "your question"   # pick a topology
"""
from __future__ import annotations

import argparse

from voyager.topologies import TOPOLOGIES

DEFAULT_Q = (
    "What is the most recent Llama model Meta has released, and what are its "
    "headline benchmark results?"
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="voyager")
    parser.add_argument("question", nargs="*", help="question to answer")
    parser.add_argument(
        "-t",
        "--topology",
        default="react",
        choices=sorted(TOPOLOGIES),
        help="agent topology to use",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() or DEFAULT_Q
    topo = TOPOLOGIES[args.topology]()

    print(f"\n[{args.topology}] Q: {question}\n")
    r = topo.run(question)
    print(f"--- Answer [{r.topology}] ---\n{r.answer}\n")
    print(
        f"steps={r.steps}  tools={r.tool_calls}  "
        f"tokens(in/out)={r.input_tokens}/{r.output_tokens}  "
        f"latency={r.latency_s:.1f}s"
    )


if __name__ == "__main__":
    main()
