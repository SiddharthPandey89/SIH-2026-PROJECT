"""
backend/model_router/router.py

Model router for the Sovereign AI Workbench.

Role of this module:
    - Given a task_type (from backend/model_router/task_classifier.py) and a
      message/history/context, pick the best available local model via
      backend/model_router/model_registry.py, call it through a pluggable
      backend adapter, and return a structured result.
    - Own conversation-turn prompt assembly (folding history + retrieved
      context into a messages list) and safe fallback behaviour when a task
      type is unknown or a model is unreachable.

Explicitly OUT of scope for this module:
    - Model/task metadata -- lives entirely in model_registry.py. This file
      never hardcodes a model name or a task-type -> model mapping; every
      selection decision goes through the registry's public API.
    - Task classification -- lives entirely in task_classifier.py.
    - Actually loading model weights in-process -- see TransformersAdapter,
      which is a clearly-marked extension point, not a real implementation.

Contract with backend/api/chat.py (not modified by this file):
    model_router: ModelRouter = Depends(get_model_router)
    generation = await model_router.generate(
        message=message, task_type=task_type, history=history,
        context_chunks=source_chunks or None,
    )
    answer = generation["answer"]
    model_used = generation["model"]
    ...
    model_router_ready = await model_router.health_check()

`generate()` therefore returns a plain, dict-subscriptable object (a
GenerationResult TypedDict) -- not a Pydantic model -- so chat.py's existing
`generation["answer"]` / `generation["model"]` access keeps working
unmodified.

Local/offline-first: every adapter talks only to a local endpoint taken
from ModelConfig.endpoint (Ollama/vLLM/llama.cpp running on the same
machine or internal network). No cloud provider is contacted anywhere in
this file. A lightweight, best-effort guard rejects obviously-external
endpoints as defense in depth; the authoritative enforcement is the
infrastructure-level network guard (backend/security/network_guard.py).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import httpx

from backend.model_router.model_registry import (
    DEFAULT_TASK_TYPE,
    ModelBackend,
    ModelConfig,
    ModelNotFoundError,
    ModelRegistry,
    get_model_registry,
)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS = 3.0

# Best-effort, defense-in-depth guard: hostnames that should never appear in
# a ModelConfig.endpoint. Not the primary enforcement mechanism -- that is
# backend/security/network_guard.py at the OS/network level -- but catching
# an obviously-external endpoint here fails fast with a clear error instead
# of silently leaking a prompt off-premises.
_BLOCKED_ENDPOINT_SUBSTRINGS = (
    "openai.com",
    "anthropic.com",
    "googleapis.com",
    "azure.com",
    "amazonaws.com",
    "cohere.ai",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class GenerationResult(TypedDict):
    """Return shape of ModelRouter.generate(); a plain dict at runtime."""

    answer: str
    model: str
    backend: str
    task_type: str
    fallback_used: bool


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RouterError(Exception):
    """Base exception for router-level failures."""


class ExternalEndpointBlockedError(RouterError):
    """Raised when a model's configured endpoint looks like a public cloud address."""


class AdapterNotConfiguredError(RouterError):
    """Raised when no inference adapter is registered for a model's backend."""


class NoModelAvailableError(RouterError):
    """Raised when every candidate model for a task (including fallback) failed or none exist."""


# ---------------------------------------------------------------------------
# Inference adapters
# ---------------------------------------------------------------------------
# One adapter per local serving engine. Adding a new engine later means
# adding one adapter class and one registry entry in _build_default_adapters
# -- ModelRouter itself never changes.


class InferenceAdapter(ABC):
    """Common interface every local backend engine must implement."""

    @abstractmethod
    async def generate(self, model: ModelConfig, messages: List[Dict[str, str]], timeout: float) -> str:
        """Send messages to the local model server and return the assistant's reply text."""

    @abstractmethod
    async def health_check(self, model: ModelConfig, timeout: float) -> bool:
        """Return True if this model's local backend is currently reachable and responsive."""


class OllamaAdapter(InferenceAdapter):
    """Adapter for a local Ollama server (chat endpoint)."""

    async def generate(self, model: ModelConfig, messages: List[Dict[str, str]], timeout: float) -> str:
        url = f"{model.endpoint.rstrip('/')}/api/chat"
        payload = {"model": model.backend_model_name, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RouterError(f"Unexpected response shape from Ollama for model '{model.model_id}'.") from exc

    async def health_check(self, model: ModelConfig, timeout: float) -> bool:
        url = f"{model.endpoint.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.HTTPError:
            return False


class OpenAICompatibleAdapter(InferenceAdapter):
    """
    Adapter for local engines that expose an OpenAI-compatible /v1 API --
    covers vLLM and modern llama.cpp server builds. Shared implementation
    since both speak the same wire format.
    """

    def __init__(self, health_path: str = "/v1/models") -> None:
        self._health_path = health_path

    async def generate(self, model: ModelConfig, messages: List[Dict[str, str]], timeout: float) -> str:
        url = f"{model.endpoint.rstrip('/')}/v1/chat/completions"
        payload = {"model": model.backend_model_name, "messages": messages}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RouterError(
                f"Unexpected response shape from OpenAI-compatible backend for model '{model.model_id}'."
            ) from exc

    async def health_check(self, model: ModelConfig, timeout: float) -> bool:
        url = f"{model.endpoint.rstrip('/')}{self._health_path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.HTTPError:
            return False


class TransformersAdapter(InferenceAdapter):
    """
    Placeholder for in-process (no HTTP server) inference via the
    `transformers` library -- relevant for vision/embedding models that may
    be loaded directly in the backend process rather than served over HTTP.

    Intentionally NOT implemented here: loading real weights is out of
    scope for this file. This class exists so ModelBackend.TRANSFORMERS is
    a first-class, addressable backend in the adapter map -- wiring in a
    real implementation later requires no changes to ModelRouter.
    """

    async def generate(self, model: ModelConfig, messages: List[Dict[str, str]], timeout: float) -> str:
        raise NotImplementedError(
            f"In-process transformers inference is not yet implemented for model '{model.model_id}'."
        )

    async def health_check(self, model: ModelConfig, timeout: float) -> bool:
        return False


def _build_default_adapters() -> Dict[ModelBackend, InferenceAdapter]:
    return {
        ModelBackend.OLLAMA: OllamaAdapter(),
        ModelBackend.VLLM: OpenAICompatibleAdapter(health_path="/v1/models"),
        ModelBackend.LLAMACPP: OpenAICompatibleAdapter(health_path="/v1/models"),
        ModelBackend.TRANSFORMERS: TransformersAdapter(),
    }


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_messages(
    message: str,
    history: Optional[List[Dict[str, str]]],
    context_chunks: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """
    Assemble an OpenAI/Ollama-style messages list from the current turn's
    inputs. Kept intentionally simple -- this is orchestration-level
    formatting, not prompt engineering or RAG strategy (that judgement, if
    it grows more elaborate, belongs in knowledge_base/retriever.py).
    """
    messages: List[Dict[str, str]] = []

    if context_chunks:
        context_block = "\n\n".join(
            f"[{chunk.get('title', 'Untitled')}] {chunk.get('snippet', '')}" for chunk in context_chunks
        )
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following retrieved context from the organization's knowledge base "
                    "to help answer the user, if relevant. Do not fabricate information beyond it.\n\n"
                    f"{context_block}"
                ),
            }
        )

    for turn in history or []:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})
    return messages


def _assert_local_endpoint(model: ModelConfig) -> None:
    """Best-effort guard against an obviously-external endpoint. See module docstring."""
    endpoint_lower = model.endpoint.lower()
    for blocked in _BLOCKED_ENDPOINT_SUBSTRINGS:
        if blocked in endpoint_lower:
            raise ExternalEndpointBlockedError(
                f"Model '{model.model_id}' has endpoint '{model.endpoint}', which looks like a public "
                "cloud API. Refusing to call it -- this workbench is local/offline-first only."
            )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """
    Selects a local model for a task type via ModelRegistry, calls it
    through the appropriate InferenceAdapter, and falls back safely when
    the requested task type or the top-choice model is unavailable.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        adapters: Optional[Dict[ModelBackend, InferenceAdapter]] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = registry or get_model_registry()
        self._adapters = adapters or _build_default_adapters()
        self._request_timeout = request_timeout
        self._health_check_timeout = health_check_timeout

    def _get_adapter(self, backend: ModelBackend) -> InferenceAdapter:
        adapter = self._adapters.get(backend)
        if adapter is None:
            raise AdapterNotConfiguredError(f"No inference adapter configured for backend '{backend.value}'.")
        return adapter

    def _resolve_candidates(self, task_type: str) -> tuple[List[ModelConfig], bool]:
        """
        Return (candidates, fallback_used) for a task type: the registry's
        ranked candidates for task_type, or -- if none exist, e.g. an
        unrecognized/unsupported task type -- the ranked candidates for
        DEFAULT_TASK_TYPE instead.
        """
        candidates = self._registry.get_models_for_task(task_type)
        if candidates:
            return candidates, False

        logger.warning(
            "No enabled model supports task_type='%s'; falling back to default task_type='%s'.",
            task_type,
            DEFAULT_TASK_TYPE,
        )
        fallback_candidates = self._registry.get_models_for_task(DEFAULT_TASK_TYPE)
        return fallback_candidates, True

    async def generate(
        self,
        message: str,
        task_type: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> GenerationResult:
        """
        Generate a response for one conversation turn.

        Tries candidate models for task_type in priority order, falling
        back to the default task type's models if task_type is unknown/
        unsupported, and moving on to the next candidate if a given model's
        backend is unreachable or errors. Raises NoModelAvailableError only
        if every candidate (across both the requested and fallback task
        type) fails.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a non-empty string.")

        candidates, task_fallback_used = self._resolve_candidates(task_type)
        if not candidates:
            raise NoModelAvailableError(
                f"No enabled model is available for task_type='{task_type}' or the default task type."
            )

        messages = _build_messages(message, history, context_chunks)

        last_error: Optional[Exception] = None
        for index, model in enumerate(candidates):
            try:
                _assert_local_endpoint(model)
                adapter = self._get_adapter(model.backend)
                answer = await adapter.generate(model, messages, timeout=self._request_timeout)
                return GenerationResult(
                    answer=answer,
                    model=model.model_id,
                    backend=model.backend.value,
                    task_type=task_type,
                    fallback_used=task_fallback_used or index > 0,
                )
            except ExternalEndpointBlockedError:
                # Configuration error, not a transient failure -- do not
                # silently try another model, surface it immediately.
                raise
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any backend can fail in many ways
                logger.warning(
                    "Model '%s' (backend=%s) failed to generate for task_type='%s': %s. Trying next candidate.",
                    model.model_id,
                    model.backend.value,
                    task_type,
                    exc,
                )
                last_error = exc
                continue

        raise NoModelAvailableError(
            f"All {len(candidates)} candidate model(s) for task_type='{task_type}' failed to respond."
        ) from last_error

    async def health_check(self) -> bool:
        """
        Return True if at least one enabled model's backend currently
        responds. Used by GET /api/chat/health; kept fast via a short
        per-model timeout and by checking models concurrently.
        """
        models = self._registry.list_models(enabled_only=True)
        if not models:
            logger.warning("Model router health check: no enabled models registered.")
            return False

        async def _check(model: ModelConfig) -> bool:
            try:
                adapter = self._get_adapter(model.backend)
                return await adapter.health_check(model, timeout=self._health_check_timeout)
            except Exception:  # noqa: BLE001 - a single bad adapter must not fail the whole check
                logger.debug("Health check failed for model '%s'.", model.model_id, exc_info=True)
                return False

        results = await asyncio.gather(*(_check(model) for model in models))
        return any(results)


# ---------------------------------------------------------------------------
# Module-level singleton + FastAPI-style dependency factory
# ---------------------------------------------------------------------------
#
# get_model_router() is deliberately a zero-argument callable. FastAPI
# inspects the signature of every function passed to Depends(...) to build
# its dependency graph; any bare parameter on that function without its own
# Depends()/Query()/Body() marker gets treated as a request parameter that
# FastAPI must build a validation field for. ModelRegistry is a plain
# Python class, not a Pydantic-compatible type, so a signature like
#     def get_model_router(registry: Optional[ModelRegistry] = None) -> ModelRouter: ...
# makes FastAPI try (and fail) to turn `registry` into a query/body field,
# raising "Invalid args for response field ... ModelRegistry | None" at
# route registration time -- before a single request is even handled.
#
# ModelRouter's own __init__ still accepts an optional registry/adapters
# for direct instantiation in tests; that constructor is never introspected
# by FastAPI because it isn't the callable passed to Depends().

_default_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """
    FastAPI dependency factory returning the shared ModelRouter singleton.

    Takes no parameters -- safe to use as Depends(get_model_router). Builds
    the singleton on first call using the real get_model_registry(), then
    reuses it for the lifetime of the process.
    """
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter(registry=get_model_registry())
    return _default_router