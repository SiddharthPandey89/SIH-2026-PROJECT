"""
backend/agent/memory.py

Runtime memory for the Agent Engine.

Responsibilities:
    - Store agent task context.
    - Store conversation messages used during an agent task.
    - Store intermediate outputs produced by executed steps.
    - Store step errors/status information.
    - Provide a clean snapshot that can later be persisted by the
      database/task-log layer.

This module intentionally does NOT:
    - execute tools
    - call the LLM
    - perform database queries
    - perform vector search
    - read/write files

Persistent conversation history belongs to the database layer.
This class represents the agent's runtime working memory.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional


__all__ = [
    "MemoryError",
    "MemoryValidationError",
    "ConversationMessage",
    "StepMemory",
    "AgentMemory",
    "get_memory",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base exception for agent memory failures."""


class MemoryValidationError(MemoryError):
    """Raised when invalid data is supplied to agent memory."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ConversationMessage:
    """One conversation message stored in agent runtime memory."""

    role: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class StepMemory:
    """Runtime information about one executed agent step."""

    step_id: str
    tool: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent Memory
# ---------------------------------------------------------------------------


class AgentMemory:
    """
    Runtime working memory for one agent task.

    Memory is intentionally kept independent from SQLAlchemy/database code.
    This makes the agent engine testable without requiring a database and
    allows the database layer to persist snapshots separately.
    """

    VALID_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(
        self,
        *,
        task_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        goal: Optional[str] = None,
        task_type: Optional[str] = None,
        max_history: int = 50,
    ) -> None:
        if not isinstance(max_history, int):
            raise MemoryValidationError(
                "max_history must be an integer."
            )

        if max_history < 1:
            raise MemoryValidationError(
                "max_history must be at least 1."
            )

        self.task_id = self._clean_optional_string(
            task_id,
            "task_id",
        )

        self.conversation_id = self._clean_optional_string(
            conversation_id,
            "conversation_id",
        )

        self.goal = self._clean_optional_string(
            goal,
            "goal",
        )

        self.task_type = self._clean_optional_string(
            task_type,
            "task_type",
        )

        self.max_history = max_history

        self._context: Dict[str, Any] = {}
        self._messages: List[ConversationMessage] = []
        self._steps: List[StepMemory] = []

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_optional_string(
        value: Optional[str],
        field_name: str,
    ) -> Optional[str]:
        """Validate and normalize an optional string."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise MemoryValidationError(
                f"{field_name} must be a string or None."
            )

        value = value.strip()

        return value or None

    @staticmethod
    def _require_string(
        value: Any,
        field_name: str,
    ) -> str:
        """Validate and normalize a required non-empty string."""

        if not isinstance(value, str) or not value.strip():
            raise MemoryValidationError(
                f"{field_name} must be a non-empty string."
            )

        return value.strip()

    # -----------------------------------------------------------------------
    # Task metadata
    # -----------------------------------------------------------------------

    def set_task(
        self,
        *,
        task_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        goal: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> None:
        """
        Update task-level metadata.

        Only supplied values are changed.
        """

        if task_id is not None:
            self.task_id = self._require_string(
                task_id,
                "task_id",
            )

        if conversation_id is not None:
            self.conversation_id = self._require_string(
                conversation_id,
                "conversation_id",
            )

        if goal is not None:
            self.goal = self._require_string(
                goal,
                "goal",
            )

        if task_type is not None:
            self.task_type = self._require_string(
                task_type,
                "task_type",
            )

    # -----------------------------------------------------------------------
    # Context
    # -----------------------------------------------------------------------

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Set or replace one context value.

        Example:
            memory.set_context("file_id", "abc123")
            memory.set_context("document_name", "report.pdf")
        """

        key = self._require_string(
            key,
            "context key",
        )

        self._context[key] = deepcopy(value)

    def update_context(
        self,
        values: Mapping[str, Any],
    ) -> None:
        """
        Add or replace multiple context values.
        """

        if not isinstance(values, Mapping):
            raise MemoryValidationError(
                "context values must be a mapping."
            )

        for key, value in values.items():
            key = self._require_string(
                key,
                "context key",
            )

            self._context[key] = deepcopy(value)

    def get_context(
        self,
        key: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """
        Return one context value or the complete context.

        If key is provided:
            Return the value associated with that key.

        If key is None:
            Return a defensive copy of the complete context.
        """

        if key is None:
            return deepcopy(self._context)

        key = self._require_string(
            key,
            "context key",
        )

        if key not in self._context:
            return default

        return deepcopy(self._context[key])

    def get_all_context(self) -> Dict[str, Any]:
        """Return a defensive copy of the complete task context."""

        return deepcopy(self._context)

    def remove_context(
        self,
        key: str,
    ) -> bool:
        """
        Remove one context value.

        Returns:
            True if the key existed, otherwise False.
        """

        key = self._require_string(
            key,
            "context key",
        )

        if key not in self._context:
            return False

        del self._context[key]

        return True

    # -----------------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
    ) -> ConversationMessage:
        """
        Add a conversation message to runtime memory.

        Supported roles:
            system
            user
            assistant
            tool
        """

        role = self._require_string(
            role,
            "role",
        ).lower()

        if role not in self.VALID_ROLES:
            raise MemoryValidationError(
                f"Invalid message role '{role}'. "
                f"Expected one of: {sorted(self.VALID_ROLES)}."
            )

        content = self._require_string(
            content,
            "content",
        )

        message = ConversationMessage(
            role=role,
            content=content,
        )

        self._messages.append(message)

        # Keep only the most recent messages.
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

        return message

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Return conversation history in model-router-compatible format.

        Output:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        """

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self._messages
        ]

    def get_message_objects(
        self,
    ) -> List[ConversationMessage]:
        """Return copies of the internal message objects."""

        return deepcopy(self._messages)

    def clear_messages(self) -> None:
        """Clear only conversation messages."""

        self._messages.clear()

    # -----------------------------------------------------------------------
    # Step / intermediate output memory
    # -----------------------------------------------------------------------

    def record_step(
        self,
        *,
        step_id: str,
        tool: str,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
    ) -> StepMemory:
        """
        Record the result of an executed agent step.

        This is the main place where executor output can be stored.
        """

        step_id = self._require_string(
            step_id,
            "step_id",
        )

        tool = self._require_string(
            tool,
            "tool",
        )

        if not isinstance(success, bool):
            raise MemoryValidationError(
                "success must be a boolean."
            )

        if error is not None:
            if not isinstance(error, str):
                raise MemoryValidationError(
                    "error must be a string or None."
                )

            error = error.strip() or None

        step = StepMemory(
            step_id=step_id,
            tool=tool,
            success=success,
            output=deepcopy(output),
            error=error,
        )

        self._steps.append(step)

        return step

    def get_step(
        self,
        step_id: str,
    ) -> Optional[StepMemory]:
        """
        Return the most recent recorded step with the given ID.
        """

        step_id = self._require_string(
            step_id,
            "step_id",
        )

        for step in reversed(self._steps):
            if step.step_id == step_id:
                return deepcopy(step)

        return None

    def get_step_output(
        self,
        step_id: str,
        default: Any = None,
    ) -> Any:
        """
        Return the output of the most recent step with the given ID.
        """

        step = self.get_step(step_id)

        if step is None:
            return default

        return deepcopy(step.output)

    def get_steps(self) -> List[StepMemory]:
        """Return copies of all recorded step results."""

        return deepcopy(self._steps)

    def get_outputs(self) -> Dict[str, Any]:
        """
        Return successful step outputs indexed by step ID.

        Example:
            {
                "step_1": "...",
                "step_2": {"result": 123}
            }
        """

        outputs: Dict[str, Any] = {}

        for step in self._steps:
            if step.success:
                outputs[step.step_id] = deepcopy(
                    step.output
                )

        return outputs

    def clear_steps(self) -> None:
        """Clear intermediate step memory."""

        self._steps.clear()

    # -----------------------------------------------------------------------
    # Combined state
    # -----------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current runtime memory state.

        This is useful for:
            - debugging
            - API responses
            - task logging
            - future database persistence
        """

        return {
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "goal": self.goal,
            "task_type": self.task_type,
            "context": self.get_all_context(),
            "messages": self.get_messages(),
            "steps": [
                step.to_dict()
                for step in self._steps
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias for get_state()."""

        return self.get_state()

    # -----------------------------------------------------------------------
    # State restoration
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "AgentMemory":
        """
        Reconstruct AgentMemory from a previously generated
        state dictionary.

        This does not perform database I/O.
        """

        if not isinstance(data, Mapping):
            raise MemoryValidationError(
                "Memory state must be a mapping."
            )

        memory = cls(
            task_id=data.get("task_id"),
            conversation_id=data.get("conversation_id"),
            goal=data.get("goal"),
            task_type=data.get("task_type"),
        )

        context = data.get(
            "context",
            {},
        )

        if context is not None:
            memory.update_context(context)

        messages = data.get(
            "messages",
            [],
        )

        if not isinstance(messages, list):
            raise MemoryValidationError(
                "messages must be a list."
            )

        for item in messages:
            if not isinstance(item, Mapping):
                raise MemoryValidationError(
                    "Each message must be a mapping."
                )

            memory.add_message(
                role=item.get("role"),
                content=item.get("content"),
            )

        steps = data.get(
            "steps",
            [],
        )

        if not isinstance(steps, list):
            raise MemoryValidationError(
                "steps must be a list."
            )

        for item in steps:
            if not isinstance(item, Mapping):
                raise MemoryValidationError(
                    "Each step must be a mapping."
                )

            memory.record_step(
                step_id=item.get("step_id"),
                tool=item.get("tool"),
                success=item.get("success"),
                output=item.get("output"),
                error=item.get("error"),
            )

        return memory

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def clear(self) -> None:
        """
        Clear runtime memory while keeping task metadata.

        Task metadata such as task_id and conversation_id remains intact.
        """

        self._context.clear()
        self._messages.clear()
        self._steps.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_memory(
    *,
    task_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    goal: Optional[str] = None,
    task_type: Optional[str] = None,
    max_history: int = 50,
) -> AgentMemory:
    """
    Create a new AgentMemory instance.

    A new instance is intentionally returned on every call.

    Agent tasks must not share runtime memory with each other.
    """

    return AgentMemory(
        task_id=task_id,
        conversation_id=conversation_id,
        goal=goal,
        task_type=task_type,
        max_history=max_history,
    )