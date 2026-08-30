"""
backend/model_router/model_registry.py

Central metadata registry for local, open-weight models used by the
Sovereign AI Workbench.

Role of this module:
    - Define a type-safe, immutable description of each available local
      model (where it lives, how it's served, what tasks it's good for).
    - Provide lookup APIs that task_classifier.py and router.py will use to
      pick a model for a given task, without either of them needing to know
      about specific models, ports, or backends directly.
    - Allow new open-weight models to be added later purely by registering
      another ModelConfig -- no changes required in task_classifier.py or
      router.py.

Explicitly OUT of scope for this module (by design):
    - Loading model weights or running inference.
    - Talking to Ollama/vLLM/llama.cpp over HTTP.
    - Health checks against a running server.
    These belong in router.py once implemented; this module is pure
    metadata + lookup.

Everything here is local-first: endpoints default to localhost, and
model_registry never itself makes a network call.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelBackend(str, Enum):
    """Which local serving engine hosts a given model."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"  # in-process, no HTTP server


class Modality(str, Enum):
    """What kind of input/output a model primarily handles."""

    TEXT = "text"
    VISION = "vision"
    EMBEDDING = "embedding"


# ---------------------------------------------------------------------------
# Known task types
# ---------------------------------------------------------------------------
# Kept as plain strings rather than a strict Enum, matching backend/api/schemas.py:
# the set of task types is expected to grow as new models/capabilities are
# added, and this registry should not need editing just because a new task
# type string appears elsewhere in the system.

TASK_CHAT = "chat"
TASK_CODE = "code"
TASK_DOCUMENT_QA = "document_qa"
TASK_SUMMARIZATION = "summarization"
TASK_SPREADSHEET = "spreadsheet"
TASK_VISION = "vision"
TASK_GENERAL = "general"

DEFAULT_TASK_TYPE = TASK_CHAT


# ---------------------------------------------------------------------------
# Model configuration schema
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """
    Static description of a single local model. Instances are immutable --
    changing a model's config means registering a new ModelConfig, not
    mutating one in place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        ...,
        description="Unique key used to reference this model elsewhere in the system.",
        examples=["qwen2.5-7b"],
    )
    display_name: str = Field(..., description="Human-readable name for UI/logs.")
    backend: ModelBackend = Field(..., description="Local serving engine that hosts this model.")
    modality: Modality = Field(default=Modality.TEXT, description="Primary input/output modality.")

    endpoint: str = Field(
        ...,
        description="Base URL of the local serving engine (e.g. http://localhost:11434). "
        "Always local/offline -- never a public internet address.",
    )
    backend_model_name: str = Field(
        ...,
        description="Identifier the serving engine itself uses for this model "
        "(e.g. an Ollama tag like 'qwen2.5:7b').",
    )
    weights_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to the model weights, if applicable "
        "(e.g. 'models/llm/qwen2.5-7b').",
    )

    supported_tasks: List[str] = Field(
        ...,
        min_length=1,
        description="Task types this model is suitable for, e.g. ['chat', 'document_qa'].",
    )
    context_window: int = Field(..., gt=0, description="Maximum context length in tokens.")
    quantization: Optional[str] = Field(default=None, description="Quantization scheme, if any (e.g. 'Q4_K_M').")
    approx_vram_gb: Optional[float] = Field(
        default=None, ge=0.0, description="Approximate VRAM footprint in GB, for capacity planning."
    )

    priority: int = Field(
        default=100,
        description="Selection priority within a task type; lower value is preferred first. "
        "Used to break ties when multiple models support the same task.",
    )
    enabled: bool = Field(default=True, description="Whether this model is currently eligible for selection.")
    notes: Optional[str] = Field(default=None, description="Free-text notes (e.g. known limitations).")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model_id is not registered."""


class DuplicateModelError(ModelRegistryError):
    """Raised when attempting to register a model_id that already exists without overwrite=True."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    In-memory registry of available local models.

    task_classifier.py / router.py should interact with this class (or the
    module-level singleton below) rather than hardcoding model names.
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelConfig] = {}

    # -- registration ---------------------------------------------------

    def register(self, config: ModelConfig, overwrite: bool = False) -> None:
        """Add a model to the registry. New open-weight models are added this way."""
        if config.model_id in self._models and not overwrite:
            raise DuplicateModelError(
                f"Model '{config.model_id}' is already registered. Pass overwrite=True to replace it."
            )
        self._models[config.model_id] = config
        logger.info(
            "Registered model '%s' (%s) for tasks=%s",
            config.model_id,
            config.backend.value,
            config.supported_tasks,
        )

    def unregister(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def set_enabled(self, model_id: str, enabled: bool) -> None:
        """Toggle availability without removing the model's config."""
        current = self.get(model_id)
        self._models[model_id] = current.model_copy(update={"enabled": enabled})

    # -- lookup -----------------------------------------------------------

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelNotFoundError(f"No model registered with id '{model_id}'.") from exc

    def list_models(self, enabled_only: bool = True) -> List[ModelConfig]:
        models = list(self._models.values())
        if enabled_only:
            models = [m for m in models if m.enabled]
        return sorted(models, key=lambda m: (m.priority, m.model_id))

    def get_models_for_task(self, task_type: str, enabled_only: bool = True) -> List[ModelConfig]:
        """
        Return models capable of handling task_type, best candidate first
        (lowest priority value first, then model_id for stable ordering).
        """
        candidates = [m for m in self._models.values() if task_type in m.supported_tasks]
        if enabled_only:
            candidates = [m for m in candidates if m.enabled]
        return sorted(candidates, key=lambda m: (m.priority, m.model_id))

    def get_default_model_for_task(self, task_type: str) -> ModelConfig:
        """
        Return the single best model for task_type, falling back to the best
        model for DEFAULT_TASK_TYPE if nothing supports task_type directly.
        """
        candidates = self.get_models_for_task(task_type)
        if candidates:
            return candidates[0]

        logger.warning(
            "No enabled model supports task_type='%s'; falling back to default task_type='%s'.",
            task_type,
            DEFAULT_TASK_TYPE,
        )
        fallback_candidates = self.get_models_for_task(DEFAULT_TASK_TYPE)
        if fallback_candidates:
            return fallback_candidates[0]

        raise ModelNotFoundError(
            f"No enabled model supports task_type='{task_type}' or the default task_type='{DEFAULT_TASK_TYPE}'."
        )

    def all_supported_task_types(self, enabled_only: bool = True) -> List[str]:
        """Union of task types supported by all (enabled) registered models, sorted."""
        models = self.list_models(enabled_only=enabled_only)
        task_types = {task for model in models for task in model.supported_tasks}
        return sorted(task_types)


# ---------------------------------------------------------------------------
# Default local endpoints (overridable via environment variables)
# ---------------------------------------------------------------------------

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Seed data: models available in the local Ollama installation
# ---------------------------------------------------------------------------


def _build_default_models() -> List[ModelConfig]:
    return [
        ModelConfig(
            model_id="qwen2.5-7b",
            display_name="Qwen 2.5 7B",
            backend=ModelBackend.OLLAMA,
            modality=Modality.TEXT,
            endpoint=_OLLAMA_HOST,
            backend_model_name="qwen2.5:7b",
            weights_path="models/llm/qwen2.5-7b",
            supported_tasks=[
                TASK_CHAT,
                TASK_CODE,
                TASK_DOCUMENT_QA,
                TASK_SUMMARIZATION,
                TASK_GENERAL,
                TASK_SPREADSHEET,
            ],
            context_window=32768,
            quantization=None,
            approx_vram_gb=5.0,
            priority=10,
            notes="Primary general-purpose local model. Also used for coding and document tasks.",
        ),
        ModelConfig(
            model_id="qwen2.5-3b",
            display_name="Qwen 2.5 3B",
            backend=ModelBackend.OLLAMA,
            modality=Modality.TEXT,
            endpoint=_OLLAMA_HOST,
            backend_model_name="qwen2.5:3b",
            weights_path="models/llm/qwen2.5-3b",
            supported_tasks=[
                TASK_CHAT,
                TASK_CODE,
                TASK_DOCUMENT_QA,
                TASK_SUMMARIZATION,
                TASK_GENERAL,
                TASK_SPREADSHEET,
            ],
            context_window=32768,
            quantization=None,
            approx_vram_gb=2.5,
            priority=20,
            notes="Lightweight fallback model for general, coding and document tasks.",
        ),
        ModelConfig(
            model_id="llama3.2-1b",
            display_name="Llama 3.2 1B",
            backend=ModelBackend.OLLAMA,
            modality=Modality.TEXT,
            endpoint=_OLLAMA_HOST,
            backend_model_name="llama3.2:1b",
            weights_path="models/llm/llama3.2-1b",
            supported_tasks=[
                TASK_CHAT,
                TASK_GENERAL,
                TASK_SUMMARIZATION,
            ],
            context_window=8192,
            quantization=None,
            approx_vram_gb=1.5,
            priority=30,
            notes="Small lightweight fallback model for simple text tasks.",
        ),
        ModelConfig(
            model_id="llama3.2-latest",
            display_name="Llama 3.2 Latest",
            backend=ModelBackend.OLLAMA,
            modality=Modality.TEXT,
            endpoint=_OLLAMA_HOST,
            backend_model_name="llama3.2:latest",
            weights_path="models/llm/llama3.2-latest",
            supported_tasks=[
                TASK_CHAT,
                TASK_GENERAL,
                TASK_SUMMARIZATION,
            ],
            context_window=8192,
            quantization=None,
            approx_vram_gb=2.0,
            priority=40,
            notes="Additional local fallback model for general text tasks.",
        ),
    ]


def _register_default_models(registry: ModelRegistry) -> None:
    for config in _build_default_models():
        registry.register(config)


# ---------------------------------------------------------------------------
# Module-level singleton + FastAPI-style dependency factory
# ---------------------------------------------------------------------------

_default_registry = ModelRegistry()
_register_default_models(_default_registry)


def get_model_registry() -> ModelRegistry:
    """
    Dependency factory returning the shared registry instance. Mirrors the
    get_model_router() / get_retriever() pattern used elsewhere in the API layer.
    """
    return _default_registry


def reset_registry_to_defaults(registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """
    Rebuild a registry with only the default seed models. Primarily useful
    for tests that register temporary/mock models and need a clean slate.
    """
    target = registry if registry is not None else _default_registry
    target._models.clear()
    _register_default_models(target)
    return target