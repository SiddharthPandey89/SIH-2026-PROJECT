"""
In-process Mistral 7B engine.

The model router calls this through adapter.py. This module owns loading,
generation, streaming, health, and shutdown. It never performs routing or
task classification.
"""

from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:
    from .exceptions import MistralGenerationError, MistralNotLoadedError
    from .generation import GenerationParams, resolve_generation_params
    from .loader import load_backend, validate_model_files
    from .prompt import normalize_messages
    from .settings import MistralSettings, load_settings
except ImportError:
    from exceptions import MistralGenerationError, MistralNotLoadedError
    from generation import GenerationParams, resolve_generation_params
    from loader import load_backend, validate_model_files
    from prompt import normalize_messages
    from settings import MistralSettings, load_settings

logger = logging.getLogger(__name__)

_ENGINE_LOCK = threading.RLock()
_ENGINES: Dict[str, "Mistral7BEngine"] = {}
_ATEXIT_REGISTERED = False


class Mistral7BEngine:
    """Process-local Mistral 7B runtime bound to one weights path."""

    def __init__(self, settings: Optional[MistralSettings] = None, model_path: Optional[str] = None) -> None:
        self._settings = settings or load_settings(model_path=model_path)
        self._backend = None
        self._load_lock = threading.RLock()
        self._closed = False

    @property
    def model_path(self) -> Path:
        return self._settings.model_path

    @property
    def loaded(self) -> bool:
        return self._backend is not None and not self._closed

    def initialize(self) -> None:
        """Load weights into memory. Safe to call more than once."""
        if self._closed:
            raise MistralNotLoadedError("This Mistral 7B engine has been shut down.")
        with self._load_lock:
            if self._backend is not None:
                return
            logger.info("Initializing Mistral 7B from %s", self._settings.model_path)
            self._backend = load_backend(self._settings)
            logger.info(
                "Mistral 7B ready (backend=%s, device=%s)",
                self._backend.name,
                self._backend.device,
            )

    def health(self) -> Dict[str, Any]:
        """
        Status for the router health check.

        Does not load weights. Reports whether the configured path is valid
        and whether this process already has the model in memory.
        """
        try:
            backend_name, artifact = validate_model_files(
                self._settings.model_path,
                self._settings.preferred_backend,
            )
            return {
                "ok": True,
                "loaded": self.loaded,
                "model_path": str(self._settings.model_path),
                "artifact_path": str(artifact),
                "backend": self._backend.name if self._backend is not None else backend_name,
                "device": self._backend.device if self._backend is not None else self._settings.device,
                "closed": self._closed,
            }
        except Exception as exc:
            logger.warning("Mistral 7B health check failed: %s", exc)
            return {
                "ok": False,
                "loaded": False,
                "model_path": str(self._settings.model_path),
                "backend": None,
                "device": self._settings.device,
                "closed": self._closed,
                "error": str(exc),
            }

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Return a complete assistant reply for a chat-messages list."""
        return "".join(
            self.generate_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        ).strip()

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper for a single user prompt."""
        return self.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def generate_stream(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """Yield reply text as it is produced. Compatible extras for future SSE."""
        self.initialize()
        if self._backend is None:
            raise MistralNotLoadedError("Mistral 7B failed to initialize.")

        normalized = normalize_messages(messages)
        if not normalized:
            raise MistralGenerationError("Cannot generate from an empty messages list.")

        params: GenerationParams = resolve_generation_params(
            self._settings,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        try:
            yield from self._backend.generate_stream(normalized, params)
        except MistralGenerationError:
            raise
        except Exception as exc:
            raise MistralGenerationError(f"Mistral 7B generation failed: {exc}") from exc

    def shutdown(self) -> None:
        """Release GPU/CPU resources. Subsequent generate() calls will fail."""
        with self._load_lock:
            if self._backend is not None:
                logger.info("Shutting down Mistral 7B engine at %s", self._settings.model_path)
                try:
                    self._backend.close()
                finally:
                    self._backend = None
            self._closed = True


def get_engine(model_path: Optional[str] = None) -> Mistral7BEngine:
    """Return a process-wide engine for the given (or default) weights path."""
    global _ATEXIT_REGISTERED
    settings = load_settings(model_path=model_path)
    key = str(settings.model_path)
    with _ENGINE_LOCK:
        engine = _ENGINES.get(key)
        if engine is None or engine._closed:
            engine = Mistral7BEngine(settings=settings)
            _ENGINES[key] = engine
        if not _ATEXIT_REGISTERED:
            atexit.register(shutdown_all)
            _ATEXIT_REGISTERED = True
        return engine


def shutdown_all() -> None:
    with _ENGINE_LOCK:
        engines: List[Mistral7BEngine] = list(_ENGINES.values())
        _ENGINES.clear()
    for engine in engines:
        try:
            engine.shutdown()
        except Exception:
            logger.debug("Error while shutting down a Mistral 7B engine.", exc_info=True)
