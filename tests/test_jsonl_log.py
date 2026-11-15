import json
from pathlib import Path

import pytest

from jsonl_log import JsonlLog


def test_roundtrip_and_lookup(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    JsonlLog(p).extend([{"key": "a", "v": 1}, {"v": 2}])
    log = JsonlLog(p)
    assert len(log.rows) == 2 and log.get("a") == {"key": "a", "v": 1}


def test_torn_last_line_is_dropped_and_file_cleaned(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b", "x"', encoding="utf-8")
    log = JsonlLog(p)
    assert [r["key"] for r in log.rows] == ["a"]
    log.append({"key": "c"})
    assert [r["key"] for r in JsonlLog(p).rows] == ["a", "c"]


def test_corrupt_middle_line_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    p.write_text('{"key": "a"}\nnot json\n{"key": "b"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _ = JsonlLog(p)


def test_orphan_children_are_dropped_from_memory_and_file(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    rows = [
        {"kind": "candidate", "belongs_to": "s1", "i": 1},
        {"kind": "candidate", "belongs_to": "s1", "i": 2},
        {"kind": "candidate", "belongs_to": "s2", "i": 1},
        {"kind": "result", "key": "s1"},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log = JsonlLog(p)
    assert len(log.rows) == 3 and all(r["belongs_to"] == "s1" for r in log.rows[:2])
    assert len(p.read_text(encoding="utf-8").splitlines()) == 3


def test_duplicate_key_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    log = JsonlLog(p)
    log.append({"key": "a"})
    with pytest.raises(ValueError):
        log.append({"key": "a"})
