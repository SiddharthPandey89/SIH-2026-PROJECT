"""
Router-facing adapter for local Mistral 7B.

Matches backend.model_router.router.InferenceAdapter (async generate +
health_check) without importing the router, so this directory stays free of
routing logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

try:
    from .engine import Mistral7BEngine, get_engine
except ImportError:
    from engine import Mistral7BEngine, get_engine


class Mistral7BInferenceAdapter:
    """
    Duck-typed InferenceAdapter.

    ModelRouter calls:
        await adapter.generate(model, messages, timeout)
        await adapter.health_check(model, timeout)
    """

    def __init__(self, engine: Optional[Mistral7BEngine] = None) -> None:
        self._engine = engine

    def _engine_for(self, model: Any) -> Mistral7BEngine:
        if self._engine is not None:
            return self._engine
        weights_path = getattr(model, "weights_path", None)
        return get_engine(model_path=weights_path)

    async def generate(self, model: Any, messages: List[Dict[str, str]], timeout: float) -> str:
        engine = self._engine_for(model)
        return await asyncio.wait_for(
            asyncio.to_thread(engine.generate, messages),
            timeout=timeout,
        )

    async def generate_stream(self, model: Any, messages: List[Dict[str, str]], timeout: float):
        """
        Async wrapper around token streaming.

        The current ModelRouter.generate() contract returns a single string,
        so the router does not call this. It is available for a future
        streaming route without changing engine internals.
        """
        engine = self._engine_for(model)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        sentinel = None

        def _produce() -> None:
            try:
                for chunk in engine.generate_stream(messages):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        worker = asyncio.create_task(asyncio.to_thread(_produce))
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await worker

    async def health_check(self, model: Any, timeout: float) -> bool:
        engine = self._engine_for(model)

        def _check() -> bool:
            return bool(engine.health().get("ok"))

        try:
            return await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout)
        except Exception:
            return False
