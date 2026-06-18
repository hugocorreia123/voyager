"""Topology contract.

Every topology returns a RunResult with the same instrumentation, so the eval
harness (Phase 3) can compare them on quality, cost, and latency without caring
how each one works internally. Getting this contract right now is what makes the
five-topology comparison clean later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RunResult:
    topology: str
    question: str
    answer: str
    steps: int = 0                       # LLM calls made
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    trace: list[str] = field(default_factory=list)  # message-type sequence

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Topology(ABC):
    """Base class for all five agent topologies."""

    name: str

    @abstractmethod
    def run(self, question: str) -> RunResult:
        """Answer a question and return an instrumented result."""
        raise NotImplementedError
