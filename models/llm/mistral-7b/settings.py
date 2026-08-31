"""
Local configuration for the in-process Mistral 7B engine.

Values come from environment variables, with filesystem defaults relative to
the repository root. Nothing here contacts a network service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_COMPONENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _COMPONENT_DIR.parents[2]
DEFAULT_WEIGHTS_DIR = _COMPONENT_DIR / "weights"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_optional_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@dataclass(frozen=True)
class MistralSettings:
    """Resolved runtime settings for one Mistral 7B engine instance."""

    model_path: Path
    device: str
    dtype: str
    load_in_4bit: bool
    load_in_8bit: bool
    n_ctx: int
    n_gpu_layers: int
    n_threads: Optional[int]
    preferred_backend: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool

    @classmethod
    def from_env(cls, model_path: Optional[str] = None) -> "MistralSettings":
        raw_path = model_path or os.getenv("MISTRAL_MODEL_PATH") or str(DEFAULT_WEIGHTS_DIR)
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        device = (os.getenv("MISTRAL_DEVICE") or "auto").strip().lower()
        if device not in {"auto", "cuda", "cpu"}:
            device = "auto"

        dtype = (os.getenv("MISTRAL_DTYPE") or "auto").strip().lower()
        preferred = (os.getenv("MISTRAL_BACKEND") or "auto").strip().lower()
        if preferred not in {"auto", "transformers", "llamacpp"}:
            preferred = "auto"

        n_threads_raw = os.getenv("MISTRAL_N_THREADS")
        n_threads = int(n_threads_raw) if n_threads_raw and n_threads_raw.strip() else None

        temperature = _env_float("MISTRAL_TEMPERATURE", 0.7)
        max_tokens = _env_int("MISTRAL_MAX_TOKENS", 512)
        top_p = _env_float("MISTRAL_TOP_P", 0.9)
        top_k = _env_int("MISTRAL_TOP_K", 50)
        repetition_penalty = _env_float("MISTRAL_REPETITION_PENALTY", 1.1)

        return cls(
            model_path=path.resolve(),
            device=device,
            dtype=dtype,
            load_in_4bit=_env_bool("MISTRAL_LOAD_IN_4BIT", False),
            load_in_8bit=_env_bool("MISTRAL_LOAD_IN_8BIT", False),
            n_ctx=_env_int("MISTRAL_N_CTX", 8192),
            n_gpu_layers=_env_int("MISTRAL_N_GPU_LAYERS", -1),
            n_threads=n_threads,
            preferred_backend=preferred,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
        )


def load_settings(model_path: Optional[str] = None) -> MistralSettings:
    return MistralSettings.from_env(model_path=model_path)
