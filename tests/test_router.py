"""
tests/test_router.py

Unit tests for backend.model_router.

These tests are intentionally offline:
- No Ollama server required
- No vLLM server required
- No llama.cpp server required
- No real LLM weights required
- No internet required

The real ModelRouter, ModelRegistry and adapter contracts are tested
using fake local adapters.
"""

import pytest

from backend.model_router.router import (
    ModelRouter,
    RouterError,
    ExternalEndpointBlockedError,
    AdapterNotConfiguredError,
    NoModelAvailableError,
    InferenceAdapter,
    _build_messages,
    _assert_local_endpoint,
)

from backend.model_router.model_registry import (
    ModelBackend,
    ModelConfig,
    ModelRegistry,
    ModelNotFoundError,
    DuplicateModelError,
    TASK_CHAT,
    TASK_CODE,
    TASK_DOCUMENT_QA,
    TASK_SUMMARIZATION,
    TASK_VISION,
)


# ============================================================
# Fake Adapter
# ============================================================

class FakeAdapter(InferenceAdapter):
    """
    Deterministic adapter used instead of a real model server.
    """

    def __init__(
        self,
        answer="Fake model response",
        healthy=True,
        fail=False,
    ):
        self.answer = answer
        self.healthy = healthy
        self.fail = fail

        self.generate_calls = []
        self.health_calls = []

    async def generate(
        self,
        model,
        messages,
        timeout,
    ):
        self.generate_calls.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
            }
        )

        if self.fail:
            raise RuntimeError(
                "Fake inference failure"
            )

        return self.answer

    async def health_check(
        self,
        model,
        timeout,
    ):
        self.health_calls.append(
            {
                "model": model,
                "timeout": timeout,
            }
        )

        return self.healthy


# ============================================================
# Helpers
# ============================================================

def make_model(
    model_id="fake-model",
    task=TASK_CHAT,
    priority=10,
    backend=ModelBackend.OLLAMA,
    endpoint="http://localhost:11434",
    enabled=True,
):
    return ModelConfig(
        model_id=model_id,
        display_name=model_id,
        backend=backend,
        modality="text",
        endpoint=endpoint,
        backend_model_name=model_id,
        weights_path=None,
        supported_tasks=[task],
        context_window=4096,
        quantization=None,
        approx_vram_gb=1.0,
        priority=priority,
        enabled=enabled,
        notes="Test model",
    )


def make_router(
    models,
    adapters,
):
    registry = ModelRegistry()

    for model in models:
        registry.register(model)

    return ModelRouter(
        registry=registry,
        adapters=adapters,
        request_timeout=5.0,
        health_check_timeout=1.0,
    )


# ============================================================
# Message Assembly Tests
# ============================================================

def test_build_messages_basic():
    """
    Basic request should produce a single user message.
    """

    messages = _build_messages(
        message="Hello",
        history=None,
        context_chunks=None,
    )

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


def test_build_messages_with_history():
    """
    Conversation history should be preserved before the
    current user message.
    """

    history = [
        {
            "role": "user",
            "content": "What is a pump?",
        },
        {
            "role": "assistant",
            "content": "A pump moves fluid.",
        },
    ]

    messages = _build_messages(
        message="Explain it again.",
        history=history,
        context_chunks=None,
    )

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is a pump?"

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "A pump moves fluid."

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Explain it again."


def test_build_messages_with_context():
    """
    Retrieved KB context should be inserted as a system message.
    """

    context = [
        {
            "title": "Pump Manual",
            "snippet": "Check pump vibration before operation.",
        }
    ]

    messages = _build_messages(
        message="What should I check?",
        history=None,
        context_chunks=context,
    )

    assert messages[0]["role"] == "system"

    assert "Pump Manual" in messages[0]["content"]

    assert (
        "Check pump vibration before operation."
        in messages[0]["content"]
    )

    assert messages[-1]["role"] == "user"


def test_build_messages_ignores_empty_history_content():
    """
    Empty history entries should not be added.
    """

    history = [
        {
            "role": "user",
            "content": "",
        },
        {
            "role": "assistant",
            "content": "Useful answer",
        },
    ]

    messages = _build_messages(
        message="Next question",
        history=history,
        context_chunks=None,
    )

    assert len(messages) == 2

    assert messages[0]["content"] == "Useful answer"
    assert messages[1]["content"] == "Next question"


# ============================================================
# Local Endpoint Security Tests
# ============================================================

def test_local_endpoint_is_allowed():
    """
    Local endpoints must pass the security guard.
    """

    model = make_model(
        endpoint="http://localhost:11434"
    )

    _assert_local_endpoint(model)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
        "https://example.azure.com",
        "https://model.amazonaws.com",
        "https://api.cohere.ai",
    ],
)
def test_external_endpoint_is_blocked(endpoint):
    """
    Obvious public-cloud endpoints must be rejected.
    """

    model = make_model(
        endpoint=endpoint
    )

    with pytest.raises(
        ExternalEndpointBlockedError
    ):
        _assert_local_endpoint(model)


# ============================================================
# Model Selection Tests
# ============================================================

@pytest.mark.asyncio
async def test_router_selects_model_for_chat():
    """
    Chat task should select a model supporting chat.
    """

    model = make_model(
        model_id="chat-model",
        task=TASK_CHAT,
    )

    adapter = FakeAdapter(
        answer="Chat response"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Hello",
        task_type=TASK_CHAT,
    )

    assert result["answer"] == "Chat response"
    assert result["model"] == "chat-model"
    assert result["backend"] == "ollama"
    assert result["task_type"] == TASK_CHAT
    assert result["fallback_used"] is False


@pytest.mark.asyncio
async def test_router_selects_code_model():
    """
    Coding task should select the coding model.
    """

    model = make_model(
        model_id="deepseek-coder-test",
        task=TASK_CODE,
    )

    adapter = FakeAdapter(
        answer="Code generated"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Write Python code",
        task_type=TASK_CODE,
    )

    assert result["model"] == (
        "deepseek-coder-test"
    )

    assert result["answer"] == (
        "Code generated"
    )


@pytest.mark.asyncio
async def test_router_selects_document_model():
    """
    Document QA should use a model supporting document_qa.
    """

    model = make_model(
        model_id="document-model",
        task=TASK_DOCUMENT_QA,
    )

    adapter = FakeAdapter(
        answer="Document answer"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="What does the document say?",
        task_type=TASK_DOCUMENT_QA,
    )

    assert result["model"] == "document-model"


@pytest.mark.asyncio
async def test_router_selects_vision_model():
    """
    Vision task should select a vision-capable registered model.
    """

    model = make_model(
        model_id="vision-model",
        task=TASK_VISION,
    )

    adapter = FakeAdapter(
        answer="Vision analysis"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Analyze this image",
        task_type=TASK_VISION,
    )

    assert result["model"] == "vision-model"


# ============================================================
# Priority / Fallback Tests
# ============================================================

@pytest.mark.asyncio
async def test_router_uses_highest_priority_model():
    """
    Lower priority number wins.
    """

    first = make_model(
        model_id="priority-10",
        task=TASK_CHAT,
        priority=10,
    )

    second = make_model(
        model_id="priority-20",
        task=TASK_CHAT,
        priority=20,
    )

    adapter = FakeAdapter(
        answer="Primary response"
    )

    router = make_router(
        [second, first],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Hello",
        task_type=TASK_CHAT,
    )

    assert result["model"] == "priority-10"

    assert result["fallback_used"] is False


@pytest.mark.asyncio
async def test_router_falls_back_when_first_model_fails():
    """
    If the first candidate fails, router should try the next one.
    """

    first = make_model(
        model_id="first-model",
        task=TASK_CHAT,
        priority=10,
    )

    second = make_model(
        model_id="second-model",
        task=TASK_CHAT,
        priority=20,
    )

    failing_adapter = FakeAdapter(
        fail=True
    )

    working_adapter = FakeAdapter(
        answer="Fallback response"
    )

    router = make_router(
        [first, second],
        {
            ModelBackend.OLLAMA: failing_adapter,
        },
    )

    # Use a custom adapter object that changes behavior
    # based on the selected model.
    class MultiModelAdapter(InferenceAdapter):

        async def generate(
            self,
            model,
            messages,
            timeout,
        ):
            if model.model_id == "first-model":
                raise RuntimeError(
                    "First model failed"
                )

            return "Fallback response"

        async def health_check(
            self,
            model,
            timeout,
        ):
            return True

    router = make_router(
        [first, second],
        {
            ModelBackend.OLLAMA:
                MultiModelAdapter()
        },
    )

    result = await router.generate(
        message="Hello",
        task_type=TASK_CHAT,
    )

    assert result["model"] == "second-model"

    assert result["answer"] == (
        "Fallback response"
    )

    assert result["fallback_used"] is True


@pytest.mark.asyncio
async def test_router_falls_back_for_unknown_task():
    """
    Unknown task should fall back to the default chat task.
    """

    model = make_model(
        model_id="default-chat",
        task=TASK_CHAT,
    )

    adapter = FakeAdapter(
        answer="Default response"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Something unusual",
        task_type="unknown_task",
    )

    assert result["model"] == "default-chat"

    assert result["fallback_used"] is True

    assert result["task_type"] == (
        "unknown_task"
    )


# ============================================================
# Input Validation Tests
# ============================================================

@pytest.mark.asyncio
async def test_empty_message_is_rejected():
    """
    Empty message must raise ValueError.
    """

    model = make_model()

    adapter = FakeAdapter()

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    with pytest.raises(ValueError):
        await router.generate(
            message="",
            task_type=TASK_CHAT,
        )


@pytest.mark.asyncio
async def test_whitespace_message_is_rejected():
    """
    Whitespace-only message must raise ValueError.
    """

    model = make_model()

    adapter = FakeAdapter()

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    with pytest.raises(ValueError):
        await router.generate(
            message="     ",
            task_type=TASK_CHAT,
        )


@pytest.mark.asyncio
async def test_empty_task_type_is_rejected():
    """
    Empty task type must raise ValueError.
    """

    model = make_model()

    adapter = FakeAdapter()

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    with pytest.raises(ValueError):
        await router.generate(
            message="Hello",
            task_type="",
        )


# ============================================================
# No Model Tests
# ============================================================

@pytest.mark.asyncio
async def test_no_model_available():
    """
    If neither requested task nor default chat has a model,
    router should raise NoModelAvailableError.
    """

    model = make_model(
        model_id="code-only",
        task=TASK_CODE,
    )

    adapter = FakeAdapter()

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    with pytest.raises(
        NoModelAvailableError
    ):
        await router.generate(
            message="Hello",
            task_type="unknown_task",
        )


# ============================================================
# Adapter Tests
# ============================================================

@pytest.mark.asyncio
async def test_missing_adapter_is_rejected():
    """
    If registry contains a backend but router has no adapter,
    AdapterNotConfiguredError should be raised internally and
    eventually result in NoModelAvailableError.
    """

    model = make_model(
        backend=ModelBackend.VLLM,
        endpoint="http://localhost:8000",
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA:
                FakeAdapter()
        },
    )

    with pytest.raises(
        NoModelAvailableError
    ):
        await router.generate(
            message="Hello",
            task_type=TASK_CHAT,
        )


# ============================================================
# Result Contract
# ============================================================

@pytest.mark.asyncio
async def test_generation_result_schema():
    """
    GenerationResult must expose all fields expected by the API.
    """

    model = make_model(
        model_id="schema-model"
    )

    adapter = FakeAdapter(
        answer="Schema response"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.generate(
        message="Test",
        task_type=TASK_CHAT,
    )

    assert set(result.keys()) == {
        "answer",
        "model",
        "backend",
        "task_type",
        "fallback_used",
    }

    assert isinstance(
        result["answer"],
        str,
    )

    assert isinstance(
        result["model"],
        str,
    )

    assert isinstance(
        result["backend"],
        str,
    )

    assert isinstance(
        result["task_type"],
        str,
    )

    assert isinstance(
        result["fallback_used"],
        bool,
    )


# ============================================================
# History + Context Forwarding
# ============================================================

@pytest.mark.asyncio
async def test_history_and_context_reach_adapter():
    """
    Router must pass assembled history and retrieved context
    to the inference adapter.
    """

    model = make_model()

    adapter = FakeAdapter(
        answer="Context response"
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    history = [
        {
            "role": "user",
            "content": "Previous question",
        },
        {
            "role": "assistant",
            "content": "Previous answer",
        },
    ]

    context = [
        {
            "title": "Safety Manual",
            "snippet": "Wear PPE.",
        }
    ]

    await router.generate(
        message="What PPE is required?",
        task_type=TASK_CHAT,
        history=history,
        context_chunks=context,
    )

    assert len(
        adapter.generate_calls
    ) == 1

    messages = (
        adapter.generate_calls[0]["messages"]
    )

    assert messages[0]["role"] == "system"

    assert (
        "Safety Manual"
        in messages[0]["content"]
    )

    assert (
        "Previous question"
        in messages[1]["content"]
    )

    assert messages[-1]["content"] == (
        "What PPE is required?"
    )


# ============================================================
# Health Check Tests
# ============================================================

@pytest.mark.asyncio
async def test_health_check_true_when_model_is_healthy():
    """
    Router is healthy when at least one enabled model backend
    responds successfully.
    """

    model = make_model()

    adapter = FakeAdapter(
        healthy=True
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_false_when_model_is_unhealthy():
    """
    Router should report unhealthy when all model backends fail.
    """

    model = make_model()

    adapter = FakeAdapter(
        healthy=False
    )

    router = make_router(
        [model],
        {
            ModelBackend.OLLAMA: adapter
        },
    )

    result = await router.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_health_check_false_with_no_models():
    """
    Empty registry means router is not ready.
    """

    router = make_router(
        [],
        {},
    )

    result = await router.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_health_check_ignores_one_failed_model():
    """
    One failed model must not make the complete router unhealthy
    if another enabled model is healthy.
    """

    first = make_model(
        model_id="bad-model",
        priority=10,
    )

    second = make_model(
        model_id="good-model",
        priority=20,
    )

    class HealthAdapter(InferenceAdapter):

        async def generate(
            self,
            model,
            messages,
            timeout,
        ):
            return "response"

        async def health_check(
            self,
            model,
            timeout,
        ):
            return model.model_id == (
                "good-model"
            )

    router = make_router