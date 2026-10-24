"""Run prompts through a defended pipeline and log structured rows."""

from __future__ import annotations

from collections.abc import Iterable

from attacks import Attack
from data import PromptRow
from defenses import Defense
from jsonl_log import JsonlLog, JsonValue, Row
from judge import Judge, RefusalStringJudge
from model import Target


class Runner:
    """Sends prompts through a defense and the model, labels each response and logs it."""

    def __init__(
        self,
        target: Target,
        log: JsonlLog,
        run_id: str,
        judge: Judge | None = None,
        max_new_tokens: int = 128,
        search_log: JsonlLog | None = None,
    ) -> None:
        """Set the model, the query log, the run id, the judge, and the optional search log."""
        self.target: Target = target
        self.log: JsonlLog = log
        self.search_log: JsonlLog | None = search_log
        self.run_id: str = run_id
        self.judge: Judge = judge or RefusalStringJudge()
        self.max_new_tokens: int = max_new_tokens

    def query(
        self,
        defense: Defense,
        row: PromptRow,
        attack: str = "none",
        arm: str = "direct",
        suffix: str = "",
        attack_queries: int = 0,
    ) -> Row:
        """Generate one response, label it and append it to the log (skipped if already logged)."""
        key = f"{self.run_id}|{defense.name}|{attack}|{arm}|{row.prompt_id}"
        cached = self.log.get(key)
        if cached is not None:
            return cached
        prompt = Attack.with_suffix(row.prompt, suffix)
        out = self.target.generate(
            defense.messages(prompt), max_new_tokens=self.max_new_tokens
        )
        rec: Row = {
            "key": key,
            "run_id": self.run_id,
            "prompt_id": row.prompt_id,
            "split": row.split,
            "defense": defense.name,
            "attack": attack,
            "arm": arm,
            "attack_queries": attack_queries,
            "prompt": prompt,
            "response": out.text,
            "refusal_label": self.judge.is_refusal(out.text),
            "attack_success": (
                self.judge.harmful_success(out.text) if row.split == "harmful" else None
            ),
            "utility_correct": (
                self.judge.utility_correct(out.text, row.answer)
                if row.split == "utility" and row.answer is not None
                else None
            ),
            "human_label": None,
            "latency_s": round(out.latency_s, 4),
            "tokens_in": out.tokens_in,
            "tokens_out": out.tokens_out,
        }
        self.log.append(rec)
        return rec

    def run_split(self, defense: Defense, prompts: Iterable[PromptRow]) -> list[Row]:
        """Run every prompt through `defense` with no attack and return the logged rows."""
        return [self.query(defense, r) for r in prompts]

    def search(self, attack: Attack, defense: Defense, row: PromptRow) -> Row:
        """Run one attack search against `defense` and log every scored query (cached on resume)."""
        if self.search_log is None:
            raise ValueError("Runner needs a search_log to run attack searches")
        key = f"{self.run_id}|search|{attack.name}|{defense.name}|{row.prompt_id}|{attack.budget}"
        cached = self.search_log.get(key)
        if cached is not None:
            return cached
        res = attack.run(defense, row.prompt)
        trace: list[JsonValue] = [
            {"query_idx": t.query_idx, "logprob": t.logprob, "best": t.best}
            for t in res.trace
        ]
        rec: Row = {
            "key": key,
            "run_id": self.run_id,
            "prompt_id": row.prompt_id,
            "attack": attack.name,
            "optimized_against": defense.name,
            "budget": attack.budget,
            "seed": attack.seed,
            "suffix": res.suffix,
            "best_logprob": res.best_logprob,
            "reached_argmax": res.reached_argmax,
            "queries": res.queries,
            "latency_s": round(res.latency_s, 4),
            "trace": trace,
        }
        self.search_log.append(rec)
        return rec
