"""
Tests for backend.agent.planner.

These tests verify the planner contract without requiring a real LLM,
Ollama server, GPU model, or network connection.
"""

from __future__ import annotations

import pytest

from backend.agent.planner import (
    InvalidPlanError,
    PlanGenerationError,
    Planner,
)


class FakeModelRouter:
    """
    Minimal fake router used to test Planner independently
    from the real model backend.
    """

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    async def generate(
        self,
        message,
        task_type,
        history=None,
        context_chunks=None,
    ):
        self.calls.append(
            {
                "message": message,
                "task_type": task_type,
                "history": history,
                "context_chunks": context_chunks,
            }
        )

        return {
            "answer": self.answer,
            "model": "fake-model",
            "backend": "test",
            "task_type": task_type,
            "fallback_used": False,
        }


@pytest.mark.asyncio
async def test_create_plan_returns_valid_plan():
    """Planner should convert valid model JSON into a structured plan."""

    router = FakeModelRouter(
        """
        {
            "goal": "Calculate 25 + 25",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "code_exec",
                    "args": {
                        "code": "print(25 + 25)"
                    },
                    "continue_on_error": false
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    plan = await planner.create_plan(
        message="Calculate 25 + 25",
    )

    assert isinstance(plan, dict)

    assert plan["goal"] == "Calculate 25 + 25"

    assert "task_type" in plan
    assert "steps" in plan

    assert isinstance(plan["steps"], list)
    assert len(plan["steps"]) == 1

    step = plan["steps"][0]

    assert step["id"] == "step_1"
    assert step["tool"] == "code_exec"
    assert step["args"]["code"] == "print(25 + 25)"
    assert step["continue_on_error"] is False


@pytest.mark.asyncio
async def test_planner_calls_model_router():
    """Planner must use the existing model-router abstraction."""

    router = FakeModelRouter(
        """
        {
            "goal": "Test request",
            "steps": []
        }
        """
    )

    planner = Planner(model_router=router)

    await planner.create_plan(
        message="Test request",
    )

    assert len(router.calls) == 1

    call = router.calls[0]

    assert call["task_type"]
    assert "Test request" in call["message"]


@pytest.mark.asyncio
async def test_planner_preserves_multiple_steps():
    """Planner should preserve ordered multi-step plans."""

    router = FakeModelRouter(
        """
        {
            "goal": "Read a file and process it",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "file_rw",
                    "args": {
                        "operation": "read",
                        "path": "example.txt"
                    }
                },
                {
                    "id": "step_2",
                    "tool": "code_exec",
                    "args": {
                        "code": "print('$step_1')"
                    }
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    plan = await planner.create_plan(
        message="Read example.txt and process it",
    )

    assert len(plan["steps"]) == 2

    assert plan["steps"][0]["id"] == "step_1"
    assert plan["steps"][1]["id"] == "step_2"

    assert plan["steps"][1]["args"]["code"] == "print('$step_1')"


@pytest.mark.asyncio
async def test_planner_accepts_markdown_json():
    """
    Local models sometimes wrap JSON in markdown fences.

    Planner should still extract the JSON.
    """

    router = FakeModelRouter(
        """
        ```json
        {
            "goal": "Test markdown JSON",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "test_tool",
                    "args": {}
                }
            ]
        }
        ```
        """
    )

    planner = Planner(model_router=router)

    plan = await planner.create_plan(
        message="Test markdown JSON",
    )

    assert plan["goal"] == "Test markdown JSON"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["tool"] == "test_tool"


@pytest.mark.asyncio
async def test_planner_rejects_invalid_json():
    """Malformed model output must raise InvalidPlanError."""

    router = FakeModelRouter(
        """
        This is not valid JSON.
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(InvalidPlanError):
        await planner.create_plan(
            message="Test invalid JSON",
        )


@pytest.mark.asyncio
async def test_planner_rejects_missing_steps():
    """A plan without steps must be rejected."""

    router = FakeModelRouter(
        """
        {
            "goal": "Missing steps"
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(InvalidPlanError):
        await planner.create_plan(
            message="Missing steps",
        )


@pytest.mark.asyncio
async def test_planner_rejects_duplicate_step_ids():
    """Every step must have a unique ID."""

    router = FakeModelRouter(
        """
        {
            "goal": "Duplicate IDs",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "tool_a",
                    "args": {}
                },
                {
                    "id": "step_1",
                    "tool": "tool_b",
                    "args": {}
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(InvalidPlanError):
        await planner.create_plan(
            message="Duplicate IDs",
        )


@pytest.mark.asyncio
async def test_planner_rejects_invalid_args():
    """Step args must be a JSON object/dictionary."""

    router = FakeModelRouter(
        """
        {
            "goal": "Invalid args",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "test_tool",
                    "args": "invalid"
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(InvalidPlanError):
        await planner.create_plan(
            message="Invalid args",
        )


@pytest.mark.asyncio
async def test_planner_rejects_missing_tool():
    """Every step must specify a tool."""

    router = FakeModelRouter(
        """
        {
            "goal": "Missing tool",
            "steps": [
                {
                    "id": "step_1",
                    "args": {}
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(InvalidPlanError):
        await planner.create_plan(
            message="Missing tool",
        )


@pytest.mark.asyncio
async def test_planner_rejects_empty_message():
    """Planner should reject empty user requests."""

    router = FakeModelRouter(
        """
        {
            "goal": "Should not execute",
            "steps": []
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(ValueError):
        await planner.create_plan(
            message="",
        )

    assert len(router.calls) == 0


@pytest.mark.asyncio
async def test_planner_rejects_whitespace_message():
    """Whitespace-only requests must also be rejected."""

    router = FakeModelRouter(
        """
        {
            "goal": "Should not execute",
            "steps": []
        }
        """
    )

    planner = Planner(model_router=router)

    with pytest.raises(ValueError):
        await planner.create_plan(
            message="   ",
        )

    assert len(router.calls) == 0


@pytest.mark.asyncio
async def test_planner_handles_empty_model_response():
    """Empty model output must raise PlanGenerationError."""

    router = FakeModelRouter("")

    planner = Planner(model_router=router)

    with pytest.raises(PlanGenerationError):
        await planner.create_plan(
            message="Test empty response",
        )


@pytest.mark.asyncio
async def test_planner_handles_model_failure():
    """Model-router failures must become PlanGenerationError."""


    class FailingRouter:
        async def generate(
            self,
            message,
            task_type,
            history=None,
            context_chunks=None,
        ):
            raise RuntimeError("Fake model failure")

    planner = Planner(model_router=FailingRouter())

    with pytest.raises(PlanGenerationError):
        await planner.create_plan(
            message="Test model failure",
        )


@pytest.mark.asyncio
async def test_planner_accepts_history_and_context():
    """History and retrieved context should be included in planning."""

    router = FakeModelRouter(
        """
        {
            "goal": "Answer using context",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "document_search",
                    "args": {}
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    history = [
        {
            "role": "user",
            "content": "What is the safety procedure?",
        }
    ]

    context_chunks = [
        {
            "title": "Safety Manual",
            "snippet": "Wear required PPE before entering the area.",
        }
    ]

    await planner.create_plan(
        message="What PPE is required?",
        history=history,
        context_chunks=context_chunks,
    )

    assert len(router.calls) == 1

    planning_prompt = router.calls[0]["message"]

    assert "What PPE is required?" in planning_prompt
    assert "What is the safety procedure?" in planning_prompt
    assert "Wear required PPE" in planning_prompt


@pytest.mark.asyncio
async def test_planner_attachment_classification_flag():
    """
    has_attachment must be passed to the existing task classifier.

    This test verifies the planner can accept attachment-aware requests
    without changing the model-router interface.
    """

    router = FakeModelRouter(
        """
        {
            "goal": "Analyze uploaded document.",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "document_search",
                    "args": {}
                }
            ]
        }
        """
    )

    planner = Planner(model_router=router)

    plan = await planner.create_plan(
        message="Analyze the uploaded document.",
        has_attachment=True,
    )

    assert plan["goal"] == "Analyze uploaded document."
    assert len(plan["steps"]) == 1