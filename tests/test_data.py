import pandas as pd
import pytest

from data import PromptSetBuilder


def _fake_read(url: str) -> pd.DataFrame:
    if "HarmBench" in url:
        n = 30
        return pd.DataFrame(
            {
                "FunctionalCategory": ["standard"] * n,
                "Behavior": [f"behavior {i}" for i in range(n)],
                "BehaviorID": [f"id{i}" for i in range(n)],
                "SemanticCategory": ["illegal"] * n,
            }
        )
    n = 60
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "prompt": [f"prompt {i}" for i in range(n)],
            "type": ["t"] * n,
            "label": ["safe"] * 30 + ["unsafe"] * 30,
        }
    )


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setattr(PromptSetBuilder, "_read_csv", staticmethod(_fake_read))
    return PromptSetBuilder(1, n_harmful=3, n_safe=4, n_unsafe_contrast=2).build()


def test_split_names(built: pd.DataFrame) -> None:
    assert set(built["split"]) == {"harmful", "benign", "unsafe_contrast", "utility"}


def test_source_labels_kept(built: pd.DataFrame) -> None:
    src = built.groupby("split")["source_label"].first()
    assert src["benign"] == "safe"
    assert src["unsafe_contrast"] == "unsafe"
    assert src["harmful"] == "unsafe"


def test_rows_carry_source_label(built: pd.DataFrame) -> None:
    rows = PromptSetBuilder.rows(built)
    by_split = {r.split: r for r in rows}
    assert by_split["benign"].source_label == "safe"
    assert by_split["utility"].source_label is None
