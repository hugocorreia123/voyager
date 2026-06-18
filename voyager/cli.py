"""Run a single question through a topology and print the instrumented result.

    uv run voyager "Compare the latest quarterly earnings of NVIDIA and AMD."
    # or:
    uv run python -m voyager.cli "your question"
"""
from __future__ import annotations

import sys

from voyager.topologies.react import ReActTopology

DEFAULT_Q = (
    "What is the most recent Llama model Meta has released, and what are its "
    "headline benchmark results?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_Q
    topo = ReActTopology()

    print(f"\nQ: {question}\n")
    r = topo.run(question)

    print(f"--- Answer [{r.topology}] ---\n{r.answer}\n")
    print(
        f"steps={r.steps}  tools={r.tool_calls}  "
        f"tokens(in/out)={r.input_tokens}/{r.output_tokens}  "
        f"latency={r.latency_s:.1f}s"
    )


if __name__ == "__main__":
    main()
