"""
Application configuration loaded from environment variables.

This module is the single place for process-wide defaults (paths, model
selection knobs). It never talks to a network service.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Local Mistral 7B (in-process transformers / llama.cpp)
# ---------------------------------------------------------------------------

MISTRAL_MODEL_PATH = os.getenv(
    "MISTRAL_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "llm" / "mistral-7b" / "weights"),
)
MISTRAL_DEVICE = os.getenv("MISTRAL_DEVICE", "auto")
MISTRAL_BACKEND = os.getenv("MISTRAL_BACKEND", "auto")
MISTRAL_DTYPE = os.getenv("MISTRAL_DTYPE", "auto")
MISTRAL_TEMPERATURE = float(os.getenv("MISTRAL_TEMPERATURE", "0.7"))
MISTRAL_MAX_TOKENS = int(os.getenv("MISTRAL_MAX_TOKENS", "512"))
MISTRAL_TOP_P = float(os.getenv("MISTRAL_TOP_P", "0.9"))
MISTRAL_TOP_K = int(os.getenv("MISTRAL_TOP_K", "50"))
MISTRAL_REPETITION_PENALTY = float(os.getenv("MISTRAL_REPETITION_PENALTY", "1.1"))
MISTRAL_N_CTX = int(os.getenv("MISTRAL_N_CTX", "8192"))
MISTRAL_N_GPU_LAYERS = int(os.getenv("MISTRAL_N_GPU_LAYERS", "-1"))
