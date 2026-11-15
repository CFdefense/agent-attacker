from pathlib import Path

from attacks import RandomSearchAttack
from data import PromptRow
from defenses import NoDefense, SystemPromptDefense
from jsonl_log import JsonlLog
from run import Runner, fingerprint
from tests.fakes import FakeTarget

ROW = PromptRow("p1", "harmful", "do a bad thing", "illegal")


def make(tmp_path: Path, target: FakeTarget, **kw: int) -> Runner:
    return Runner(
        target,  # pyright: ignore[reportArgumentType]
        JsonlLog(tmp_path / "q.jsonl"),
        "t",
        search_log=JsonlLog(tmp_path / "s.jsonl"),
        **kw,
    )


def test_fingerprint_stable_and_sensitive() -> None:
    assert fingerprint({"a": 1}) == fingerprint({"a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_same_settings_resume(tmp_path: Path) -> None:
    t = FakeTarget()
    _ = make(tmp_path, t).query(NoDefense(), ROW)
    _ = make(tmp_path, t).query(NoDefense(), ROW)
    assert t.generate_calls == 1


def test_changed_revision_not_reused(tmp_path: Path) -> None:
    _ = make(tmp_path, FakeTarget("r1")).query(NoDefense(), ROW)
    t2 = FakeTarget("r2")
    _ = make(tmp_path, t2).query(NoDefense(), ROW)
    assert t2.generate_calls == 1


def test_changed_decoding_not_reused(tmp_path: Path) -> None:
    _ = make(tmp_path, FakeTarget(), max_new_tokens=96).query(NoDefense(), ROW)
    t2 = FakeTarget()
    _ = make(tmp_path, t2, max_new_tokens=64).query(NoDefense(), ROW)
    assert t2.generate_calls == 1


def test_changed_seed_not_reused(tmp_path: Path) -> None:
    _ = make(tmp_path, FakeTarget(), seed=1).query(NoDefense(), ROW)
    t2 = FakeTarget()
    _ = make(tmp_path, t2, seed=2).query(NoDefense(), ROW)
    assert t2.generate_calls == 1


def test_changed_defense_config_not_reused(tmp_path: Path) -> None:
    class Other(SystemPromptDefense):
        SAFETY_PROMPT = "different prompt"

    t = FakeTarget()
    r = make(tmp_path, t)
    _ = r.query(SystemPromptDefense(), ROW)
    _ = r.query(Other(), ROW)
    assert t.generate_calls == 2


def test_changed_suffix_not_reused(tmp_path: Path) -> None:
    t = FakeTarget()
    r = make(tmp_path, t)
    _ = r.query(NoDefense(), ROW, suffix="aaa")
    _ = r.query(NoDefense(), ROW, suffix="bbb")
    assert t.generate_calls == 2


def test_search_saves_candidates_and_budget_changes_key(tmp_path: Path) -> None:
    t = FakeTarget()
    r = make(tmp_path, t)
    rec = r.search(RandomSearchAttack(t, budget=4, seed=1), NoDefense(), ROW)  # pyright: ignore[reportArgumentType]
    cands = rec["candidates"]
    assert isinstance(cands, list) and len(cands) == 4
    assert all(isinstance(c, dict) and "suffix" in c and "logprob" in c for c in cands)
    _ = r.search(RandomSearchAttack(t, budget=5, seed=1), NoDefense(), ROW)  # pyright: ignore[reportArgumentType]
    assert r.search_log is not None and len(r.search_log.rows) == 2
