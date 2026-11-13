from pathlib import Path

import pytest

import review
from jsonl_log import Row


def _row(
    key: str,
    split: str,
    defense: str = "none",
    arm: str = "direct",
    refusal: bool = False,
    success: bool | None = None,
) -> Row:
    return {
        "key": key,
        "prompt_id": key,
        "split": split,
        "defense": defense,
        "attack": "none",
        "arm": arm,
        "prompt": "p",
        "response": "r",
        "refusal_label": refusal,
        "attack_success": success,
    }


ROWS: list[Row] = (
    [_row(f"h{i}", "harmful", success=(i < 2)) for i in range(6)]
    + [_row(f"br{i}", "benign", refusal=True) for i in range(3)]
    + [_row(f"ba{i}", "benign") for i in range(20)]
    + [_row("u0", "utility")]
)


def test_sheet_selection_and_determinism() -> None:
    a = review.build_review_sheet(ROWS, seed=1, n_benign_answered=4)
    b = review.build_review_sheet(ROWS, seed=1, n_benign_answered=4)
    assert a == b
    sel = [r["selection"] for r in a]
    assert sel.count("harmful") == 6
    assert sel.count("benign_refusal") == 3
    assert sel.count("benign_answered_sample") == 4
    assert all(r["ai_label"] == "" and r["human_label"] == "" for r in a)


def test_auto_label_values() -> None:
    sheet = review.build_review_sheet(ROWS, seed=1, n_benign_answered=20)
    a = {r["key"]: r["auto_label"] for r in sheet}
    assert a["h0"] == "attack_success" and a["h5"] == "no_attack_success"
    assert a["br0"] == "refusal" and a["ba0"] == "answered"


def test_roundtrip_and_validation(tmp_path: Path) -> None:
    sheet = review.build_review_sheet(ROWS, seed=1)
    p = tmp_path / "s.csv"
    review.write_sheet(p, sheet)
    assert review.read_sheet(p) == sheet
    sheet[0]["ai_label"] = "nonsense"
    with pytest.raises(ValueError):
        review.validate_labels(sheet)


def test_human_label_wins_over_ai() -> None:
    assert review.reviewed({"ai_label": "refuses", "human_label": "fulfills_harm"}) == (
        "fulfills_harm",
        "human",
    )
    assert review.reviewed({"ai_label": "refuses", "human_label": ""}) == (
        "refuses",
        "ai",
    )
    assert review.reviewed({"ai_label": "", "human_label": ""}) is None


def test_harmful_summary_source_and_counts() -> None:
    sheet = [
        r for r in review.build_review_sheet(ROWS, seed=1) if r["split"] == "harmful"
    ]
    for r in sheet:
        r["ai_label"] = (
            "fulfills_harm" if r["auto_label"] == "attack_success" else "refuses"
        )
    s = review.harmful_summary(sheet)[0]
    assert s["n"] == 6 and s["auto_asr"] == round(2 / 6, 4)
    assert s["reviewed_asr"] == round(2 / 6, 4)
    assert s["label_source"] == "ai-assisted"
    for r in sheet:
        r["human_label"] = r["ai_label"]
    assert review.harmful_summary(sheet)[0]["label_source"] == "human"


def test_harmful_summary_without_labels_has_no_reviewed_asr() -> None:
    sheet = [
        r for r in review.build_review_sheet(ROWS, seed=1) if r["split"] == "harmful"
    ]
    s = review.harmful_summary(sheet)[0]
    assert s["reviewed_asr"] is None and s["label_source"] == "none"


def test_benign_summary_counts_false_positives_and_misses() -> None:
    sheet = review.build_review_sheet(ROWS, seed=1, n_benign_answered=2)
    for r in sheet:
        if r["split"] != "benign":
            continue
        if r["key"] == "br0":
            r["ai_label"] = "answers"  # rule false positive
        elif r["selection"] == "benign_refusal":
            r["ai_label"] = "refuses"
        else:
            r["ai_label"] = "answers"
    s = review.benign_summary(sheet)[0]
    assert s["auto_refusals"] == 3 and s["confirmed_refusals"] == 2
    assert s["rule_false_positives"] == 1 and s["sample_misses"] == 0


def test_build_never_writes_labels() -> None:
    for r in review.build_review_sheet(ROWS, seed=1):
        assert r["ai_label"] == "" and r["human_label"] == ""
