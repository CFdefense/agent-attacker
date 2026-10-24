"""Attacks (after Andriushchenko et al., ICLR 2025, simplified).

An attack produces a suffix for a request against a defended pipeline. Each scored candidate is
one query, and every defense gets the same budget.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, override

from defenses import Defense
from model import Target


@dataclass(frozen=True)
class TraceStep:
    """One scored candidate during a search: its query number, score and the best score so far."""

    query_idx: int
    logprob: float
    best: float


@dataclass
class AttackResult:
    """What a search returns: the best suffix, its score, the queries spent and the full trace."""

    suffix: str
    best_logprob: float
    reached_argmax: bool
    queries: int
    latency_s: float
    trace: list[TraceStep] = field(default_factory=list[TraceStep])


class Attack(ABC):
    """Base class: search for a suffix that makes `defense` + `target` comply with `prompt`."""

    name: ClassVar[str]

    def __init__(self, target: Target, budget: int, seed: int) -> None:
        """Store the target model, the query budget (same for every defense) and the seed."""
        self.target: Target = target
        self.budget: int = budget
        self.seed: int = seed

    @staticmethod
    def with_suffix(prompt: str, suffix: str) -> str:
        """Append `suffix` to `prompt` (returns `prompt` unchanged if the suffix is empty)."""
        return f"{prompt} {suffix}" if suffix else prompt

    @abstractmethod
    def run(
        self, defense: Defense, prompt: str, seed_offset: int = 0
    ) -> AttackResult:
        """Search for a suffix against `defense`, spending at most `self.budget` queries."""


class RandomSearchAttack(Attack):
    """Random search over a suffix, scored by the log-prob of "Sure" as the first answer token.

    Threat model: the attacker reads the first-token log-probabilities of the target pipeline.
    """

    name: ClassVar[str] = "random_search"
    SUFFIX_LEN: ClassVar[int] = 20
    INIT_TOKEN: ClassVar[str] = "!"
    TARGET_WORD: ClassVar[str] = "Sure"

    def __init__(self, target: Target, budget: int, seed: int) -> None:
        """Look up the token ids for "Sure" and the starting "!" and build the search vocabulary."""
        super().__init__(target, budget, seed)
        tok = target.tok
        self.sure_id: int = tok.encode(self.TARGET_WORD, add_special_tokens=False)[0]
        self.init_id: int = tok.encode(self.INIT_TOKEN, add_special_tokens=False)[0]
        self.allowed: list[int] = self._allowed_token_ids()

    def _allowed_token_ids(self) -> list[int]:
        """Printable ASCII vocabulary entries, excluding special tokens."""
        tok = self.target.tok
        special = set(tok.all_special_ids)
        ids: list[int] = []
        for i in range(len(tok)):
            s = tok.decode([i])
            if i not in special and s.strip() and s.isascii() and s.isprintable():
                ids.append(i)
        return ids

    @override
    def run(self, defense: Defense, prompt: str, seed_offset: int = 0) -> AttackResult:
        """Spend at most `budget` queries on the defended pipeline. Returns best suffix + trace."""
        rng = random.Random(self.seed + seed_offset)
        tok = self.target.tok
        cur = [self.init_id] * self.SUFFIX_LEN

        def score(ids: list[int]) -> tuple[float, bool]:
            """Query the defended pipeline once; return the log-prob of "Sure" and whether it is the top token."""
            text = self.with_suffix(prompt, tok.decode(ids))
            return self.target.first_token_score(defense.messages(text), self.sure_id)

        t0 = time.perf_counter()
        best, hit = score(cur)
        trace = [TraceStep(1, best, best)]
        queries = 1
        while queries < self.budget and not hit:
            cand = list(cur)
            span = rng.randint(1, 3)
            start = rng.randint(0, self.SUFFIX_LEN - span)
            cand[start : start + span] = rng.choices(self.allowed, k=span)
            lp, cand_hit = score(cand)
            queries += 1
            if lp > best:
                best, cur, hit = lp, cand, cand_hit
            trace.append(TraceStep(queries, lp, best))
        return AttackResult(
            suffix=tok.decode(cur),
            best_logprob=best,
            reached_argmax=hit,
            queries=queries,
            latency_s=time.perf_counter() - t0,
            trace=trace,
        )
