from metrics import Metrics


def test_utility_column_is_answer_match() -> None:
    rows = [
        {
            "split": "utility",
            "defense": "none",
            "attack": "none",
            "arm": "direct",
            "refusal_label": False,
            "utility_correct": True,
            "latency_s": 1.0,
            "attack_queries": 0,
        }
    ]
    out = Metrics(rows).summary()[0]
    assert out["answer_match_acc"] == 1.0
    assert "utility_acc" not in out
    assert "auto_asr" in out and "auto_refusal_rate" in out
    assert "asr" not in out and "refusal_rate" not in out
