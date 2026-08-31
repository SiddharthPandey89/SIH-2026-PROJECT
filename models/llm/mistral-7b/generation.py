"""Generation parameter validation for the local Mistral 7B engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from .settings import MistralSettings
    from .exceptions import MistralConfigError
except ImportError:
    from settings import MistralSettings
    from exceptions import MistralConfigError


@dataclass(frozen=True)
class GenerationParams:
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool

    def transformers_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
        return kwargs

    def llamacpp_kwargs(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repetition_penalty,
        }


def resolve_generation_params(
    settings: MistralSettings,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> GenerationParams:
    temp = settings.temperature if temperature is None else float(temperature)
    tokens = settings.max_tokens if max_tokens is None else int(max_tokens)
    p = settings.top_p if top_p is None else float(top_p)
    k = settings.top_k if top_k is None else int(top_k)
    penalty = settings.repetition_penalty if repetition_penalty is None else float(repetition_penalty)

    if tokens <= 0:
        raise MistralConfigError("max_tokens must be a positive integer.")
    if temp < 0.0:
        raise MistralConfigError("temperature must be >= 0.")
    if not 0.0 < p <= 1.0:
        raise MistralConfigError("top_p must be in (0, 1].")
    if k < 0:
        raise MistralConfigError("top_k must be >= 0.")
    if penalty <= 0.0:
        raise MistralConfigError("repetition_penalty must be > 0.")

    return GenerationParams(
        temperature=temp,
        max_tokens=tokens,
        top_p=p,
        top_k=k,
        repetition_penalty=penalty,
        do_sample=temp > 0.0,
    )
