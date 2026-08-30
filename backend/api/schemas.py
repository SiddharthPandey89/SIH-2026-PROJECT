"""
backend/api/schemas.py

Pydantic request/response models for the Chat API (backend/api/chat.py).

These are pure data-contract definitions: no I/O, no business logic, no
imports from model_router / knowledge_base / database. Keeping this file
dependency-free is what lets every other layer (routes, model router,
retriever, UI) import it without pulling in the rest of the stack.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class BaseSchema(BaseModel):
    """Common config for all schemas in this module."""

    model_config = ConfigDict(
        from_attributes=True,   # allows building from ORM rows / objects with attributes
        str_strip_whitespace=True,
        extra="forbid",         # reject unexpected fields instead of silently dropping them
    )


# ---------------------------------------------------------------------------
# Known task types
# ---------------------------------------------------------------------------
# Kept as plain strings (not a strict Enum) on purpose: the task classifier
# and model registry are expected to evolve independently of this schema as
# new open-weight models / task categories are added. This constant is a
# reference for known values and for API documentation, not a hard
# validation constraint.

KNOWN_TASK_TYPES: List[str] = [
    "chat",
    "code",
    "document_qa",
    "summarization",
    "spreadsheet",
    "vision",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatRequest(BaseSchema):
    """Payload for POST /api/chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The user's message for this turn.",
        examples=["Summarize the key findings in the attached inspection report."],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID to continue. Omit to start a new conversation.",
        examples=["b3f1c2e4-1a2b-4c3d-9e8f-123456789abc"],
    )
    file_id: Optional[str] = Field(
        default=None,
        description="ID of a previously uploaded file (scanned PDF, drawing, image, spreadsheet) "
        "to ground this message in, if relevant.",
        examples=["file_20260828_0007"],
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty or whitespace-only.")
        return value.strip()

    @field_validator("conversation_id", "file_id")
    @classmethod
    def empty_string_becomes_none(cls, value: Optional[str]) -> Optional[str]:
        # Treat "" the same as omitted, so clients that send blank strings
        # instead of nulls don't accidentally trigger lookups for an empty id.
        if value is not None and not value.strip():
            return None
        return value


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class SourceReference(BaseSchema):
    """A single retrieved knowledge-base chunk cited in a chat response."""

    document_id: str = Field(..., description="Identifier of the source document.")
    title: str = Field(..., description="Human-readable title of the source document.")
    snippet: str = Field(..., description="The retrieved passage supporting the answer.")
    score: float = Field(
        ...,
        ge=0.0,
        description="Relevance/similarity score assigned by the retriever; higher means more relevant.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ChatResponse(BaseSchema):
    """Response body for POST /api/chat."""

    answer: str = Field(..., description="The assistant's reply.")
    model: str = Field(..., description="Identifier of the local model that generated the answer.")
    task_type: str = Field(
        ...,
        description=f"Classified task type for this turn. Common values: {', '.join(KNOWN_TASK_TYPES)}.",
    )
    conversation_id: str = Field(..., description="ID of the conversation this turn belongs to.")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Knowledge-base sources used to ground the answer, if any.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the response was generated.",
    )


class ChatHealthStatus(BaseSchema):
    """Response body for GET /api/chat/health."""

    status: str = Field(
        ...,
        description="Overall pipeline status: 'ok' if all dependencies are ready, 'degraded' otherwise.",
        examples=["ok", "degraded"],
    )
    model_router_ready: bool = Field(..., description="Whether the local model router responded healthy.")
    knowledge_base_ready: bool = Field(..., description="Whether the local knowledge base/retriever responded healthy.")
    database_ready: bool = Field(..., description="Whether the database connection responded healthy.")
    checked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the health check was performed.",
    )
