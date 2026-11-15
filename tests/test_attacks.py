import pytest

from attacks import RandomSearchAttack
from defenses import NoDefense
from tests.fakes import FakeTarget


def test_every_candidate_saved_with_score() -> None:
    res = RandomSearchAttack(FakeTarget(), budget=6, seed=1).run(NoDefense(), "p")  # pyright: ignore[reportArgumentType]
    assert len(res.trace) == 6
    assert res.trace[0].suffix == "b" * 20
    assert all(
        isinstance(t.suffix, str) and isinstance(t.logprob, float) for t in res.trace
    )


def test_stops_when_sure_is_argmax() -> None:
    res = RandomSearchAttack(FakeTarget(improve_at=3), budget=50, seed=1).run(  # pyright: ignore[reportArgumentType]
        NoDefense(), "p"
    )
    assert res.queries == 3
    assert res.reached_argmax


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _ = RandomSearchAttack(FakeTarget(), budget=0, seed=1)  # pyright: ignore[reportArgumentType]


def test_same_seed_gives_same_search() -> None:
    a = RandomSearchAttack(FakeTarget(), budget=8, seed=5).run(NoDefense(), "p")  # pyright: ignore[reportArgumentType]
    b = RandomSearchAttack(FakeTarget(), budget=8, seed=5).run(NoDefense(), "q")  # pyright: ignore[reportArgumentType]
    assert [t.suffix for t in a.trace] == [t.suffix for t in b.trace]


def test_multi_token_target_word_is_rejected() -> None:
    class Two(FakeTarget):
        def __init__(self) -> None:
            super().__init__()
            self.tok.encode = lambda text, add_special_tokens=False: [1, 2]  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(ValueError):
        _ = RandomSearchAttack(Two(), budget=3, seed=1)  # pyright: ignore[reportArgumentType]


def test_config_lists_search_settings() -> None:
    cfg = RandomSearchAttack(FakeTarget(), budget=7, seed=9).config()  # pyright: ignore[reportArgumentType]
    assert cfg["budget"] == 7 and cfg["seed"] == 9 and cfg["suffix_len"] == 20
