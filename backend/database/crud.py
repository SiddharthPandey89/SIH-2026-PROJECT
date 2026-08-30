"""
database/crud.py

Async CRUD layer for chat conversation history, matching exactly what
backend/api/chat.py needs:

    conversation = await crud.get_conversation(db, conversation_id)
    history = await crud.get_conversation_history(db, conversation_id)
    conversation = await crud.create_conversation(db)
    await crud.add_message(db, conversation_id, role="user", content=message)
    await crud.add_message(db, conversation_id, role="assistant", content=answer,
                            metadata={"model": model_used, "task_type": task_type})
    database_ready = await crud.ping(db)

Explicitly OUT of scope for this module:
    - Table/column definitions -- that's database/db_models.py.
    - Engine/session creation -- that's database/init_db.py.
    - Any functionality routes_chat.py doesn't actually call (no update/
      delete/list-all-conversations helpers, etc.) -- only the five
      functions above are implemented here.

Session/commit convention: each function here owns the commit for the
write(s) it performs, per the contract documented in
database/init_db.py's get_db(); get_db() only rolls back on an unhandled
exception and always closes the session, it never commits on your behalf.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db_models import Conversation, Message, MessageRole

logger = logging.getLogger(__name__)

__all__ = [
    "CrudError",
    "ConversationNotFoundError",
    "InvalidRoleError",
    "create_conversation",
    "get_conversation",
    "get_conversation_history",
    "add_message",
    "ping",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CrudError(Exception):
    """Base exception for this module's CRUD failures."""


class ConversationNotFoundError(CrudError):
    """Raised when writing to a conversation_id that doesn't exist."""


class InvalidRoleError(CrudError):
    """Raised when add_message() is given a role that isn't a valid MessageRole."""


# ---------------------------------------------------------------------------
# Conversation operations
# ---------------------------------------------------------------------------


async def create_conversation(db: AsyncSession, title: Optional[str] = None) -> Conversation:
    """
    Create and persist a new, empty conversation.

    Returns the created Conversation (with its generated UUID `id` already
    populated) so the caller can do `conversation_id = str(conversation.id)`.
    """
    conversation = Conversation(title=title)
    db.add(conversation)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to create a new conversation.")
        raise
    await db.refresh(conversation)
    logger.info("Created conversation '%s'.", conversation.id)
    return conversation


async def get_conversation(db: AsyncSession, conversation_id: str) -> Optional[Conversation]:
    """
    Look up a conversation by id.

    Returns None both when conversation_id is missing/invalid and when no
    matching row exists -- routes_chat.py treats both cases identically
    (404), so no exception is raised for a simple "not found".
    """
    if not conversation_id or not isinstance(conversation_id, str):
        logger.debug("get_conversation called with an empty/invalid conversation_id.")
        return None

    try:
        return await db.get(Conversation, conversation_id)
    except SQLAlchemyError:
        logger.exception("Failed to look up conversation '%s'.", conversation_id)
        raise


async def get_conversation_history(db: AsyncSession, conversation_id: str) -> List[Dict[str, str]]:
    """
    Return this conversation's messages in chronological order as
    [{"role": ..., "content": ...}, ...], ready to pass straight into
    model_router.generate(history=...).

    Returns an empty list if conversation_id is missing/invalid or the
    conversation has no messages yet -- it does not raise for "no history".
    """
    if not conversation_id or not isinstance(conversation_id, str):
        logger.debug("get_conversation_history called with an empty/invalid conversation_id.")
        return []

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )
    try:
        result = await db.execute(stmt)
    except SQLAlchemyError:
        logger.exception("Failed to load history for conversation '%s'.", conversation_id)
        raise

    messages = result.scalars().all()
    return [message.to_history_entry() for message in messages]


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------


async def add_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Message:
    """
    Persist one message turn.

    `metadata["model"]` and `metadata["task_type"]` (if present) are mapped
    to Message.model / Message.task_type respectively; any other keys are
    kept together in Message.extra_metadata. Both metadata fields are
    typically only meaningful for assistant turns; passing metadata=None
    (the default, as routes_chat.py does for the user turn) leaves all
    three columns unset.
    """
    if not conversation_id or not isinstance(conversation_id, str):
        raise ValueError("conversation_id must be a non-empty string.")
    if not isinstance(content, str) or not content:
        raise ValueError("content must be a non-empty string.")

    try:
        role_enum = MessageRole(role)
    except ValueError as exc:
        valid_roles = ", ".join(r.value for r in MessageRole)
        raise InvalidRoleError(f"Invalid role '{role}'; expected one of: {valid_roles}.") from exc

    conversation = await get_conversation(db, conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(f"Cannot add a message: conversation '{conversation_id}' does not exist.")

    metadata = metadata or {}
    model = metadata.get("model")
    task_type = metadata.get("task_type")
    extra_metadata = {k: v for k, v in metadata.items() if k not in ("model", "task_type")} or None

    message = Message(
        conversation_id=conversation_id,
        role=role_enum,
        content=content,
        model=model,
        task_type=task_type,
        extra_metadata=extra_metadata,
    )
    db.add(message)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to persist message for conversation '%s'.", conversation_id)
        raise
    await db.refresh(message)
    logger.debug("Added %s message (id=%s) to conversation '%s'.", role_enum.value, message.id, conversation_id)
    return message


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def ping(db: AsyncSession) -> bool:
    """
    Cheap connectivity check for GET /api/chat/health. Returns True/False
    rather than raising -- a failed ping is a health *signal*, not an error
    the caller needs to handle specially.
    """
    try:
        await db.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        logger.warning("Database ping failed.", exc_info=True)
        return False
