"""Fake model and tokenizer so tests run without downloading weights."""

from collections.abc import Sequence

from model import Generation, Messages


class FakeTok:
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self) -> int:
        return 30

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] if text == "Sure" else [2]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(96 + int(i)) if int(i) else "" for i in ids)


class FakeTarget:
    """Scores strictly decrease, except on call `improve_at`, which scores +3.0 and is a hit."""

    def __init__(self, revision: str = "r1", improve_at: int | None = None) -> None:
        self.model_id = "fake"
        self.revision = revision
        self.dtype = "float32"
        self.tok = FakeTok()
        self.generate_calls = 0
        self.score_calls = 0
        self.improve_at = improve_at

    def generate(self, messages: Messages, max_new_tokens: int = 128) -> Generation:
        self.generate_calls += 1
        return Generation(
            text="I'm sorry, I cannot help.", tokens_in=3, tokens_out=5, latency_s=0.01
        )

    def first_token_score(
        self, messages: Messages, token_id: int
    ) -> tuple[float, bool]:
        self.score_calls += 1
        if self.score_calls == self.improve_at:
            return 3.0, True
        return -float(self.score_calls), False
