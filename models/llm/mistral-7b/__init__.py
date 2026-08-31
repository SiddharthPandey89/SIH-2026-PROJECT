"""
Local, offline Mistral 7B component.

Public surface used by backend.model_router.router.TransformersAdapter:
    get_engine(), Mistral7BEngine, Mistral7BInferenceAdapter
"""

from __future__ import annotations

from .adapter import Mistral7BInferenceAdapter
from .engine import Mistral7BEngine, get_engine, shutdown_all
from .exceptions import (
    MistralConfigError,
    MistralError,
    MistralGenerationError,
    MistralModelLoadError,
    MistralModelPathError,
    MistralNotLoadedError,
)
from .settings import MistralSettings, load_settings

__all__ = [
    "Mistral7BEngine",
    "Mistral7BInferenceAdapter",
    "MistralConfigError",
    "MistralError",
    "MistralGenerationError",
    "MistralModelLoadError",
    "MistralModelPathError",
    "MistralNotLoadedError",
    "MistralSettings",
    "get_engine",
    "load_settings",
    "shutdown_all",
]
