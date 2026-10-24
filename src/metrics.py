"""Summaries computed from the per-query log rows."""

from __future__ import annotations

from collections.abc import Sequence

from jsonl_log import Row


class Metrics:
    """Groups logged rows by (split, defense, attack, arm) and computes the PD1 measures."""

    GROUP_KEYS: tuple[str, ...] = ("split", "defense", "attack", "arm")

    def __init__(self, rows: Sequence[Row]) -> None:
        """Keep the logged rows to summarize."""
        self.rows: list[Row] = list(rows)

    @staticmethod
    def _rate(values: list[bool]) -> float | None:
        """Fraction of True values, or None if there are none."""
        return round(sum(values) / len(values), 4) if values else None

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        """Mean of the values, or None if there are none."""
        return round(sum(values) / len(values), 4) if values else None

    @staticmethod
    def _bools(rows: list[Row], field: str) -> list[bool]:
        """The boolean values of `field` across the rows (rows where it is missing are skipped)."""
        return [v for r in rows if isinstance(v := r.get(field), bool)]

    @staticmethod
    def _nums(rows: list[Row], field: str) -> list[float]:
        """The numeric values of `field` across the rows (rows where it is missing are skipped)."""
        return [
            float(v)
            for r in rows
            if isinstance(v := r.get(field), (int, float)) and not isinstance(v, bool)
        ]

    def summary(self) -> list[Row]:
        """One row per group: n, refusal rate, ASR (harmful), utility accuracy, latency, queries.

        Read the rate columns by split: refusal_rate on `benign` is the false-refusal rate (FRR).
        """
        groups: dict[tuple[str, ...], list[Row]] = {}
        for r in self.rows:
            key = tuple(str(r.get(k)) for k in self.GROUP_KEYS)
            groups.setdefault(key, []).append(r)
        out: list[Row] = []
        for key, rows in sorted(groups.items()):
            out.append(
                {
                    **dict(zip(self.GROUP_KEYS, key, strict=True)),
                    "n": len(rows),
                    "refusal_rate": self._rate(self._bools(rows, "refusal_label")),
                    "asr": self._rate(self._bools(rows, "attack_success")),
                    "utility_acc": self._rate(self._bools(rows, "utility_correct")),
                    "mean_latency_s": self._mean(self._nums(rows, "latency_s")),
                    "mean_attack_queries": self._mean(
                        self._nums(rows, "attack_queries")
                    ),
                }
            )
        return out
