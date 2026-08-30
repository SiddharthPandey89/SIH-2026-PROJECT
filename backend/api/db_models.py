"""
database/db_models.py

SQLAlchemy 2.x ORM models for chat conversation history.

Role of this module:
    - Define the persistent shape of a Conversation and its Messages, as
      required by the operations backend/api/chat.py performs through
      database/crud.py:
        * create a conversation
        * retrieve a conversation by id
        * retrieve its messages in order, oldest first
        * store a user or assistant message, with optional model/task_type
          metadata for assistant turns

Explicitly OUT of scope for this module:
    - Engine/session creation, connection pooling, `get_db()` -- that
      belongs in database/init_db.py.
    - Queries -- that belongs in database/crud.py. This file only defines
      table structure, relationships, and simple data-shape helpers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models in this package."""


class MessageRole(str, PyEnum):
    """Who authored a given message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Conversation(Base):
    """A single chat conversation/thread, identified by a UUID string."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        order_by="Message.created_at, Message.id",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Conversation(id={self.id!r}, title={self.title!r}, messages={len(self.messages)})"


class Message(Base):
    """A single turn (user or assistant) within a Conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )

    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, native_enum=False, length=20), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Populated for assistant turns (which model answered, what task type it
    # was routed as); left NULL for user turns. Kept as explicit, queryable
    # columns rather than folded into extra_metadata since routes_chat.py
    # always supplies exactly these two keys for assistant messages.
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    task_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default=None)

    # Open-ended slot for anything else callers want to attach later
    # (e.g. source document ids, confidence, latency) without a schema
    # migration. Mapped to the "metadata" column, but not named `metadata`
    # as a Python attribute since that name is reserved by DeclarativeBase.
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True, default=None
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def to_history_entry(self) -> Dict[str, str]:
        """
        Serialize to the exact shape database/crud.py's
        get_conversation_history() must return for each turn:
        {"role": "user" | "assistant", "content": str}
        """
        return {"role": self.role.value, "content": self.content}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        preview = (self.content[:40] + "...") if len(self.content) > 40 else self.content
        return f"Message(id={self.id}, role={self.role.value}, content={preview!r})"
