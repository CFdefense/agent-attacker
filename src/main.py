"""Run the PD1 experiment from the command line: baseline, defense, static vs adaptive attack.

    .venv/bin/python src/main.py --out results/pd1 --budget 50

Reruns resume from the JSONL logs in --out, so already-finished queries are skipped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from attacks import RandomSearchAttack
from data import PromptSetBuilder
from defenses import NoDefense, SystemPromptDefense
from jsonl_log import JsonlLog, Row
from metrics import Metrics
from model import Target
from run import Runner

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED = 613
RUN_ID = "pd1"
MAX_NEW_TOKENS = 96
N_ATTACK_BEHAVIORS = 10


class Args(argparse.Namespace):
    """Typed command-line arguments."""

    def __init__(self) -> None:
        """Set the default output folder and attack budget."""
        super().__init__()
        self.out: Path = Path("results/pd1")
        self.budget: int = 50


class Experiment:
    """The PD1 run: baseline, defense, and static versus adaptive attack, then the summary table."""

    def __init__(self, out: Path, budget: int) -> None:
        """Store the output folder and the attack budget (queries per search)."""
        self.out: Path = out
        self.budget: int = budget

    def run(self) -> None:
        """Run every condition, resuming from the saved logs, and write the summary."""
        rows = PromptSetBuilder.rows(
            PromptSetBuilder(SEED, cache=self.out / "prompts.csv").build()
        )
        target = Target(MODEL_ID, revision=REVISION)
        runner = Runner(
            target,
            JsonlLog(self.out / "queries.jsonl"),
            RUN_ID,
            max_new_tokens=MAX_NEW_TOKENS,
            search_log=JsonlLog(self.out / "searches.jsonl"),
        )
        none, sysprompt = NoDefense(), SystemPromptDefense()

        # Baseline: every prompt, undefended and with the safety system prompt.
        for d in (none, sysprompt):
            _ = runner.run_split(d, rows)

        # Attack: static suffix (found against no defense) vs adaptive (found against the defense).
        attack = RandomSearchAttack(target, budget=self.budget, seed=SEED)
        harmful = [r for r in rows if r.split == "harmful"][:N_ATTACK_BEHAVIORS]
        for row in harmful:
            static = runner.search(attack, none, row)
            adaptive = runner.search(attack, sysprompt, row)
            n_static = int(str(static["queries"]))
            n_adaptive = int(str(adaptive["queries"]))
            for d, arm, found, n in (
                (none, "static", static, n_static),
                (sysprompt, "static", static, n_static),
                (sysprompt, "adaptive", adaptive, n_adaptive),
            ):
                _ = runner.query(
                    d,
                    row,
                    attack=attack.name,
                    arm=arm,
                    suffix=str(found["suffix"]),
                    attack_queries=n,
                )

        self._write_summary(Metrics(runner.log.rows).summary())

    def _write_summary(self, summary: list[Row]) -> None:
        """Save the summary rows to summary.csv in the output folder and print them."""
        path = self.out / "summary.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Wrote {path}")
        for r in summary:
            print(r)


def main() -> None:
    """Parse the command line and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--out", type=Path, default=Path("results/pd1"))
    _ = parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args(namespace=Args())
    Experiment(args.out, args.budget).run()


if __name__ == "__main__":
    main()
