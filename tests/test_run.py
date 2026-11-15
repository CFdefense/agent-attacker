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
    assert r.search_log is not None
    rec = r.search(RandomSearchAttack(t, budget=4, seed=1), NoDefense(), ROW)  # pyright: ignore[reportArgumentType]
    cands = [x for x in r.search_log.rows if x["kind"] == "candidate"]
    assert len(cands) == 4 and "candidates" not in rec
    assert all("suffix" in c and "logprob" in c and "query_idx" in c for c in cands)
    assert [c["query_idx"] for c in cands] == [1, 2, 3, 4]
    _ = r.search(RandomSearchAttack(t, budget=5, seed=1), NoDefense(), ROW)  # pyright: ignore[reportArgumentType]
    assert len([x for x in r.search_log.rows if x["kind"] == "result"]) == 2


def test_finished_search_is_not_rerun_or_duplicated(tmp_path: Path) -> None:
    t = FakeTarget()
    attack = RandomSearchAttack(t, budget=4, seed=1)  # pyright: ignore[reportArgumentType]
    _ = make(tmp_path, t).search(attack, NoDefense(), ROW)
    calls = t.score_calls
    r2 = make(tmp_path, t)
    assert r2.search_log is not None
    _ = r2.search(attack, NoDefense(), ROW)
    assert t.score_calls == calls
    assert len(r2.search_log.rows) == 5  # 4 candidates + 1 result, nothing added


def test_query_record_fields_and_touched(tmp_path: Path) -> None:
    t = FakeTarget()
    r = make(tmp_path, t)
    util = PromptRow("u1", "utility", "2+2?", "qa", answer="4")
    benign = PromptRow("b1", "benign", "how to kill a process", "x", None, "safe")
    h = r.query(NoDefense(), ROW)
    u = r.query(NoDefense(), util)
    b = r.query(NoDefense(), benign)
    assert h["attack_success"] is not None and h["utility_correct"] is None
    assert u["utility_correct"] is not None and u["attack_success"] is None
    assert b["attack_success"] is None and b["source_label"] == "safe"
    _ = r.query(NoDefense(), ROW)  # cached
    assert len(r.touched) == 4 and len(r.log.rows) == 3


def test_changed_prompt_text_not_reused(tmp_path: Path) -> None:
    t = FakeTarget()
    r = make(tmp_path, t)
    _ = r.query(NoDefense(), PromptRow("p1", "harmful", "text one", "c"))
    _ = r.query(NoDefense(), PromptRow("p1", "harmful", "text two", "c"))
    assert t.generate_calls == 2


def test_interrupted_search_write_leaves_no_half_search(tmp_path: Path) -> None:
    t = FakeTarget()
    attack = RandomSearchAttack(t, budget=4, seed=1)  # pyright: ignore[reportArgumentType]
    _ = make(tmp_path, t).search(attack, NoDefense(), ROW)
    path = tmp_path / "s.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # result line lost
    r2 = make(tmp_path, t)
    assert r2.search_log is not None and r2.search_log.rows == []
    _ = r2.search(attack, NoDefense(), ROW)
    assert len(r2.search_log.rows) == 5


def test_decoding_penalty_is_part_of_the_settings(tmp_path: Path) -> None:
    t = FakeTarget()
    _ = make(tmp_path, t).query(NoDefense(), ROW)
    t2 = FakeTarget()
    t2.repetition_penalty = 1.0
    _ = make(tmp_path, t2).query(NoDefense(), ROW)
    assert t2.generate_calls == 1
