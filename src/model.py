"""Target model wrapper: chat generation, first-token log-probabilities, run metadata."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Messages = list[dict[str, str]]


class Tokenizer(Protocol):
    """The slice of the Hugging Face tokenizer API this project uses."""

    all_special_ids: list[int]
    eos_token_id: int | None

    def __len__(self) -> int:
        """Vocabulary size."""
        ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Turn text into token ids."""
        ...

    def decode(
        self, token_ids: Sequence[int] | torch.Tensor, skip_special_tokens: bool = ...
    ) -> str:
        """Turn token ids back into text."""
        ...

    def apply_chat_template(
        self,
        conversation: Messages,
        add_generation_prompt: bool = ...,
        return_tensors: str = ...,
        return_dict: bool = ...,
    ) -> Mapping[str, torch.Tensor]:
        """Format chat messages with the model's chat template and tokenize them."""
        ...


class LogitsOutput(Protocol):
    """A model output that carries the next-token scores (logits)."""

    logits: torch.Tensor


class CausalLM(Protocol):
    """The slice of the Hugging Face causal LM API this project uses."""

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        """Generate a continuation of `input_ids`."""
        ...

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> LogitsOutput:
        """One forward pass; returns the scores for every position."""
        ...


@dataclass(frozen=True)
class Generation:
    """One model answer: the text, token counts and how long it took."""

    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float


class Target:
    """A local open instruct model that answers chat messages."""

    def __init__(
        self, model_id: str, revision: str | None = None, device: str | None = None
    ) -> None:
        """Load the tokenizer and model (at a pinned revision if given) onto the best available device."""
        self.model_id: str = model_id
        self.revision: str | None = revision
        self.device: str = device or self.pick_device()
        self.dtype: torch.dtype = (
            torch.float32 if self.device == "cpu" else torch.float16
        )
        t0 = time.perf_counter()
        self.tok: Tokenizer = cast(
            "Tokenizer",
            AutoTokenizer.from_pretrained(model_id, revision=revision),  # pyright: ignore[reportUnknownMemberType]
        )
        loaded = cast(
            "torch.nn.Module",
            AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                model_id, revision=revision, dtype=self.dtype
            ),
        )
        self.model: CausalLM = cast(
            "CausalLM", cast("object", loaded.to(self.device).eval())
        )
        self.load_seconds: float = time.perf_counter() - t0

    @staticmethod
    def pick_device() -> str:
        """CUDA if available, else Apple MPS, else CPU."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ids(self, messages: Messages) -> dict[str, torch.Tensor]:
        """Apply the chat template to the messages and move the tensors to the device."""
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def generate(self, messages: Messages, max_new_tokens: int = 128) -> Generation:
        """Greedy decoding. Returns text, token counts and wall-clock latency."""
        ids = self._ids(messages)
        n_in = int(ids["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids["input_ids"],
                attention_mask=ids["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        latency = time.perf_counter() - t0
        new = out[0][n_in:]
        return Generation(
            text=self.tok.decode(new, skip_special_tokens=True),
            tokens_in=n_in,
            tokens_out=int(new.shape[0]),
            latency_s=latency,
        )

    def first_token_score(
        self, messages: Messages, token_id: int
    ) -> tuple[float, bool]:
        """Log-prob of `token_id` as the first assistant token, and whether it is the argmax."""
        ids = self._ids(messages)
        with torch.no_grad():
            logits = (
                self.model(ids["input_ids"], ids["attention_mask"])
                .logits[0, -1]
                .float()
            )
        logp = torch.log_softmax(logits, dim=-1)
        return float(logp[token_id]), int(logp.argmax()) == token_id

    def gpu_memory(self) -> dict[str, float] | None:
        """Peak and reserved GPU memory in GB on CUDA, or None on other devices."""
        if self.device != "cuda":
            return None
        return {
            "max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "reserved_GB": round(torch.cuda.memory_reserved() / 1024**3, 3),
        }
