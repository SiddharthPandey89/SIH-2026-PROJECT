"""
Tests for backend.agent.memory
"""

import pytest

from backend.agent.memory import (
    AgentMemory,
    ConversationMessage,
    MemoryValidationError,
    StepMemory,
    get_memory,
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_memory_initializes_empty():
    memory = AgentMemory()

    assert memory.task_id is None
    assert memory.conversation_id is None
    assert memory.goal is None
    assert memory.task_type is None
    assert memory.get_context() == {}
    assert memory.get_messages() == []
    assert memory.get_steps() == []


def test_memory_initializes_with_metadata():
    memory = AgentMemory(
        task_id="task_123",
        conversation_id="conv_456",
        goal="Analyze report",
        task_type="document_qa",
    )

    assert memory.task_id == "task_123"
    assert memory.conversation_id == "conv_456"
    assert memory.goal == "Analyze report"
    assert memory.task_type == "document_qa"


def test_invalid_max_history():
    with pytest.raises(MemoryValidationError):
        AgentMemory(max_history=0)


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------


def test_set_task_metadata():
    memory = AgentMemory()

    memory.set_task(
        task_id="task_1",
        conversation_id="conv_1",
        goal="Summarize document",
        task_type="summarization",
    )

    assert memory.task_id == "task_1"
    assert memory.conversation_id == "conv_1"
    assert memory.goal == "Summarize document"
    assert memory.task_type == "summarization"


def test_set_task_only_updates_supplied_values():
    memory = AgentMemory(
        task_id="task_1",
        goal="Old goal",
    )

    memory.set_task(goal="New goal")

    assert memory.task_id == "task_1"
    assert memory.goal == "New goal"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_set_and_get_context():
    memory = AgentMemory()

    memory.set_context("file_id", "abc123")

    assert memory.get_context("file_id") == "abc123"


def test_get_missing_context_returns_default():
    memory = AgentMemory()

    assert memory.get_context("missing") is None
    assert memory.get_context("missing", "default") == "default"


def test_update_context():
    memory = AgentMemory()

    memory.update_context(
        {
            "file_id": "abc123",
            "filename": "report.pdf",
            "page_count": 5,
        }
    )

    assert memory.get_context("file_id") == "abc123"
    assert memory.get_context("filename") == "report.pdf"
    assert memory.get_context("page_count") == 5


def test_context_is_defensively_copied():
    memory = AgentMemory()

    original = {"pages": [1, 2, 3]}

    memory.set_context("document", original)

    original["pages"].append(4)

    assert memory.get_context("document") == {
        "pages": [1, 2, 3]
    }


def test_get_all_context_returns_copy():
    memory = AgentMemory()

    memory.set_context("key", {"value": 10})

    context = memory.get_all_context()
    context["key"]["value"] = 999

    assert memory.get_context("key") == {"value": 10}


def test_remove_context():
    memory = AgentMemory()

    memory.set_context("temporary", "value")

    assert memory.remove_context("temporary") is True
    assert memory.get_context("temporary") is None


def test_remove_missing_context():
    memory = AgentMemory()

    assert memory.remove_context("missing") is False


def test_invalid_context_key():
    memory = AgentMemory()

    with pytest.raises(MemoryValidationError):
        memory.set_context("", "value")


def test_invalid_context_mapping():
    memory = AgentMemory()

    with pytest.raises(MemoryValidationError):
        memory.update_context(["invalid"])


# ---------------------------------------------------------------------------
# Conversation messages
# ---------------------------------------------------------------------------


def test_add_message():
    memory = AgentMemory()

    message = memory.add_message(
        role="user",
        content="Analyze this document.",
    )

    assert isinstance(message, ConversationMessage)
    assert message.role == "user"
    assert message.content == "Analyze this document."


def test_get_messages():
    memory = AgentMemory()

    memory.add_message(
        role="user",
        content="Hello",
    )

    memory.add_message(
        role="assistant",
        content="Hello. How can I help?",
    )

    assert memory.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello. How can I help?",
        },
    ]


def test_supported_message_roles():
    memory = AgentMemory()

    for role in ["system", "user", "assistant", "tool"]:
        memory.add_message(
            role=role,
            content=f"Message from {role}",
        )

    messages = memory.get_messages()

    assert len(messages) == 4
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_invalid_message_role():
    memory = AgentMemory()

    with pytest.raises(MemoryValidationError):
        memory.add_message(
            role="invalid",
            content="Hello",
        )


def test_empty_message_content():
    memory = AgentMemory()

    with pytest.raises(MemoryValidationError):
        memory.add_message(
            role="user",
            content="",
        )


def test_message_history_limit():
    memory = AgentMemory(max_history=3)

    for number in range(5):
        memory.add_message(
            role="user",
            content=f"message-{number}",
        )

    messages = memory.get_messages()

    assert len(messages) == 3

    assert messages == [
        {"role": "user", "content": "message-2"},
        {"role": "user", "content": "message-3"},
        {"role": "user", "content": "message-4"},
    ]


def test_clear_messages():
    memory = AgentMemory()

    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi")

    memory.clear_messages()

    assert memory.get_messages() == []


# ---------------------------------------------------------------------------
# Step memory / intermediate outputs
#