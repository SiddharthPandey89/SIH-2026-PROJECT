"""
Tests for backend.agent.executer.

These tests verify the executor independently from the real backend tools,
LLMs, Ollama, databases, filesystem, and network.
"""

from __future__ import annotations

import pytest

from backend.agent.executer import (
    Executor,
    InvalidPlanError,
    ToolExecutionError,
    ToolNotFoundError,
)


# ============================================================================
# Fake tools
# ============================================================================


def add_tool(a: int, b: int) -> int:
    """Simple synchronous tool."""
    return a + b


def multiply_tool(value: int, multiplier: int) -> int:
    """Another synchronous tool."""
    return value * multiplier


async def async_add_tool(a: int, b: int) -> int:
    """Simple asynchronous tool."""
    return a + b


def failing_tool() -> None:
    """Tool that intentionally fails."""
    raise RuntimeError("intentional test failure")


# ============================================================================
# Basic execution
# ============================================================================


@pytest.mark.asyncio
async def test_execute_single_step():
    """Executor should successfully execute one registered tool."""

    executor = Executor(
        tools={
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Add two numbers",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 10,
                    "b": 20,
                },
                "continue_on_error": False,
            }
        ],
    }

    result = await executor.execute(plan)

    assert result.success is True
    assert len(result.results) == 1

    step = result.results[0]

    assert step.step_id == "step_1"
    assert step.tool == "add"
    assert step.success is True
    assert step.output == 30
    assert step.error is None


# ============================================================================
# Multiple steps
# ============================================================================


@pytest.mark.asyncio
async def test_execute_multiple_steps_in_order():
    """Executor should execute steps sequentially in plan order."""

    execution_order = []

    def first_tool():
        execution_order.append("first")
        return "first-result"

    def second_tool():
        execution_order.append("second")
        return "second-result"

    executor = Executor(
        tools={
            "first": first_tool,
            "second": second_tool,
        }
    )

    plan = {
        "goal": "Run two tools",
        "task_type": "chat",
        "steps": [
            {
                "id": "step_1",
                "tool": "first",
                "args": {},
            },
            {
                "id": "step_2",
                "tool": "second",
                "args": {},
            },
        ],
    }

    result = await executor.execute(plan)

    assert result.success is True

    assert execution_order == [
        "first",
        "second",
    ]

    assert len(result.results) == 2

    assert result.results[0].output == "first-result"
    assert result.results[1].output == "second-result"


# ============================================================================
# Step dependency
# ============================================================================


@pytest.mark.asyncio
async def test_execute_resolves_previous_step_reference():
    """
    Executor should resolve:

        "$step_1"

    to the output produced by step_1.
    """

    executor = Executor(
        tools={
            "add": add_tool,
            "multiply": multiply_tool,
        }
    )

    plan = {
        "goal": "Calculate using previous result",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 10,
                    "b": 5,
                },
            },
            {
                "id": "step_2",
                "tool": "multiply",
                "args": {
                    "value": "$step_1",
                    "multiplier": 2,
                },
            },
        ],
    }

    result = await executor.execute(plan)

    assert result.success is True

    assert result.results[0].output == 15
    assert result.results[1].output == 30


# ============================================================================
# Nested references
# ============================================================================


@pytest.mark.asyncio
async def test_execute_resolves_nested_references():
    """References should work inside nested dictionaries and lists."""

    received = {}

    def capture_tool(data, values):
        received["data"] = data
        received["values"] = values
        return "captured"

    executor = Executor(
        tools={
            "add": add_tool,
            "capture": capture_tool,
        }
    )

    plan = {
        "goal": "Test nested references",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 2,
                    "b": 3,
                },
            },
            {
                "id": "step_2",
                "tool": "capture",
                "args": {
                    "data": {
                        "result": "$step_1",
                    },
                    "values": [
                        "$step_1",
                        100,
                    ],
                },
            },
        ],
    }

    result = await executor.execute(plan)

    assert result.success is True

    assert received["data"]["result"] == 5
    assert received["values"] == [5, 100]


# ============================================================================
# Async tool
# ============================================================================


@pytest.mark.asyncio
async def test_execute_async_tool():
    """Executor should support asynchronous tools."""

    executor = Executor(
        tools={
            "async_add": async_add_tool,
        }
    )

    plan = {
        "goal": "Async addition",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "async_add",
                "args": {
                    "a": 7,
                    "b": 8,
                },
            }
        ],
    }

    result = await executor.execute(plan)

    assert result.success is True
    assert result.results[0].output == 15


# ============================================================================
# Missing tool
# ============================================================================


@pytest.mark.asyncio
async def test_missing_tool_fails_step():
    """Execution should fail when the requested tool is not registered."""

    executor = Executor()

    plan = {
        "goal": "Use unavailable tool",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "does_not_exist",
                "args": {},
            }
        ],
    }

    result = await executor.execute(plan)

    assert result.success is False
    assert len(result.results) == 1

    step = result.results[0]

    assert step.success is False
    assert step.tool == "does_not_exist"
    assert step.error is not None
    assert "not registered" in step.error


# ============================================================================
# Tool failure
# ============================================================================


@pytest.mark.asyncio
async def test_tool_failure_stops_execution():
    """
    By default, a failed step should stop subsequent steps.
    """

    second_called = False

    def second_tool():
        nonlocal second_called
        second_called = True
        return "should-not-run"

    executor = Executor(
        tools={
            "failing": failing_tool,
            "second": second_tool,
        }
    )

    plan = {
        "goal": "Test failure",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "failing",
                "args": {},
            },
            {
                "id": "step_2",
                "tool": "second",
                "args": {},
            },
        ],
    }

    result = await executor.execute(plan)

    assert result.success is False

    assert len(result.results) == 1

    assert result.results[0].success is False

    assert second_called is False


# ============================================================================
# Continue on error
# ============================================================================


@pytest.mark.asyncio
async def test_continue_on_error_allows_next_step():
    """
    continue_on_error=True should allow execution to continue.
    """

    executor = Executor(
        tools={
            "failing": failing_tool,
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Continue after failure",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "failing",
                "args": {},
                "continue_on_error": True,
            },
            {
                "id": "step_2",
                "tool": "add",
                "args": {
                    "a": 3,
                    "b": 4,
                },
            },
        ],
    }

    result = await executor.execute(plan)

    assert result.success is False

    assert len(result.results) == 2

    assert result.results[0].success is False
    assert result.results[1].success is True
    assert result.results[1].output == 7


# ============================================================================
# Tool registration
# ============================================================================


def test_register_and_unregister_tool():
    """Executor should allow tools to be registered and removed."""

    executor = Executor()

    assert executor.available_tools() == []

    executor.register_tool(
        "add",
        add_tool,
    )

    assert "add" in executor.available_tools()

    executor.unregister_tool("add")

    assert "add" not in executor.available_tools()


# ============================================================================
# Invalid registration
# ============================================================================


def test_register_invalid_tool_name():
    """Empty tool names should be rejected."""

    executor = Executor()

    with pytest.raises(ValueError):
        executor.register_tool(
            "",
            add_tool,
        )


def test_register_non_callable_tool():
    """A registered tool must be callable."""

    executor = Executor()

    with pytest.raises(TypeError):
        executor.register_tool(
            "invalid",
            123,
        )


# ============================================================================
# Invalid plan
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_plan_missing_steps():
    """Plan must contain a steps list."""

    executor = Executor()

    plan = {
        "goal": "Invalid plan",
        "task_type": "code",
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_plan_steps_not_list():
    """steps must be a list."""

    executor = Executor()

    plan = {
        "goal": "Invalid plan",
        "task_type": "code",
        "steps": {},
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_plan_missing_step_id():
    """Every step must contain an ID."""

    executor = Executor(
        tools={
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Invalid step",
        "task_type": "code",
        "steps": [
            {
                "tool": "add",
                "args": {
                    "a": 1,
                    "b": 2,
                },
            }
        ],
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_plan_duplicate_step_ids():
    """Step IDs must be unique."""

    executor = Executor(
        tools={
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Duplicate IDs",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 1,
                    "b": 2,
                },
            },
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 3,
                    "b": 4,
                },
            },
        ],
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_plan_missing_tool():
    """Every step must specify a tool."""

    executor = Executor()

    plan = {
        "goal": "Missing tool",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "args": {},
            }
        ],
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_plan_args_not_dictionary():
    """Step args must be a dictionary."""

    executor = Executor()

    plan = {
        "goal": "Invalid args",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": "invalid",
            }
        ],
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_invalid_continue_on_error():
    """continue_on_error must be a boolean."""

    executor = Executor()

    plan = {
        "goal": "Invalid continue flag",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {},
                "continue_on_error": "yes",
            }
        ],
    }

    with pytest.raises(InvalidPlanError):
        await executor.execute(plan)


# ============================================================================
# ExecutionResult outputs
# ============================================================================


@pytest.mark.asyncio
async def test_execution_result_outputs():
    """outputs property should contain successful step outputs."""

    executor = Executor(
        tools={
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Test outputs",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 5,
                    "b": 5,
                },
            }
        ],
    }

    result = await executor.execute(plan)

    assert result.outputs == {
        "step_1": 10,
    }


# ============================================================================
# Serialization
# ============================================================================


@pytest.mark.asyncio
async def test_execution_result_to_dict():
    """Execution result should be JSON-compatible."""

    executor = Executor(
        tools={
            "add": add_tool,
        }
    )

    plan = {
        "goal": "Test serialization",
        "task_type": "code",
        "steps": [
            {
                "id": "step_1",
                "tool": "add",
                "args": {
                    "a": 1,
                    "b": 2,
                },
            }
        ],
    }

    result = await executor.execute(plan)

    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["success"] is True
    assert isinstance(data["results"], list)

    assert data["results"][0]["step_id"] == "step_1"
    assert data["results"][0]["tool"] == "add"
    assert data["results"][0]["success"] is True
    assert data["results"][0]["output"] == 3