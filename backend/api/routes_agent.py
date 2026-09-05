"""
backend/api/routes_agent.py

Agent API controller.

Endpoints:
    POST /api/agent/run
    GET  /api/agent/status/{task_id}

Responsibilities:
    - Accept an agent task.
    - Resolve or create the conversation.
    - Load previous conversation history.
    - Retrieve optional local knowledge-base context.
    - Generate an executable plan through backend/agent/planner.py.
    - Execute that plan through backend/agent/executer.py.
    - Store runtime state through backend/agent/memory.py.
    - Persist user/assistant conversation messages through database/crud.py.
    - Expose runtime task status and execution results.

This module does NOT:
    - implement planning
    - implement tool execution
    - implement individual tools
    - directly call an LLM
    - implement vector search
    - implement database CRUD
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.agent.executer import Executor, get_executor
from backend.agent.memory import AgentMemory, get_memory
from backend.agent.planner import Planner, get_planner
from backend.database import crud
from backend.database.init_db import get_db
from backend.knowledge_base.retriever import Retriever, get_retriever


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"],
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_PLANNING = "planning"
TASK_STATUS_EXECUTING = "executing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    """
    Request body for POST /api/agent/run.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Natural-language task to execute.",
        examples=[
            "Read the uploaded inspection report and summarize the main findings."
        ],
    )

    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing conversation ID to continue. "
            "Omit this field to create a new conversation."
        ),
    )

    file_id: Optional[str] = Field(
        default=None,
        description=(
            "Previously uploaded file ID to provide local context."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Maximum number of knowledge-base context chunks to retrieve."
        ),
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "message must not be empty or whitespace-only."
            )

        return value

    @field_validator("conversation_id", "file_id")
    @classmethod
    def normalize_optional_ids(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()

        return value or None


class AgentRunResponse(BaseModel):
    """
    Immediate response returned by POST /api/agent/run.
    """

    task_id: str = Field(
        ...,
        description="Unique runtime agent task ID.",
    )

    conversation_id: str = Field(
        ...,
        description="Conversation associated with this agent task.",
    )

    status: str = Field(
        ...,
        description="Initial agent task status.",
    )

    message: str = Field(
        ...,
        description="Human-readable submission message.",
    )


class AgentStatusResponse(BaseModel):
    """
    Response returned by GET /api/agent/status/{task_id}.
    """

    task_id: str

    conversation_id: Optional[str] = None

    status: str

    message: Optional[str] = None

    task_type: Optional[str] = None

    goal: Optional[str] = None

    current_step: Optional[str] = None

    current_tool: Optional[str] = None

    plan: Optional[Dict[str, Any]] = None

    results: List[Dict[str, Any]] = Field(
        default_factory=list,
    )

    outputs: Dict[str, Any] = Field(
        default_factory=dict,
    )

    error: Optional[str] = None

    created_at: datetime

    updated_at: datetime

    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Runtime task state
# ---------------------------------------------------------------------------


@dataclass
class AgentTaskState:
    """
    Runtime state of one agent task.

    The current database schema does not contain an agent_tasks table,
    therefore this state is intentionally process-local for now.

    Persistent conversation messages are still stored through crud.py.
    """

    task_id: str

    conversation_id: Optional[str]

    message: str

    status: str = TASK_STATUS_QUEUED

    task_type: Optional[str] = None

    goal: Optional[str] = None

    current_step: Optional[str] = None

    current_tool: Optional[str] = None

    plan: Optional[Dict[str, Any]] = None

    results: List[Dict[str, Any]] = field(
        default_factory=list,
    )

    outputs: Dict[str, Any] = field(
        default_factory=dict,
    )

    error: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    completed_at: Optional[datetime] = None

    memory: Optional[AgentMemory] = field(
        default=None,
        repr=False,
    )


# ---------------------------------------------------------------------------
# Runtime task registry
# ---------------------------------------------------------------------------

_TASKS: Dict[str, AgentTaskState] = {}

_TASKS_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


async def _store_task(
    task: AgentTaskState,
) -> None:
    """Store a task in the process-local registry."""

    async with _TASKS_LOCK:
        _TASKS[task.task_id] = task


async def _get_task(
    task_id: str,
) -> Optional[AgentTaskState]:
    """Get a task from the process-local registry."""

    async with _TASKS_LOCK:
        return _TASKS.get(task_id)


async def _update_task(
    task: AgentTaskState,
) -> None:
    """Update task timestamp and store it."""

    task.updated_at = _utc_now()

    async with _TASKS_LOCK:
        _TASKS[task.task_id] = task


def _result_to_dict(
    result: Any,
) -> Dict[str, Any]:
    """
    Convert an Executor StepResult into an API-safe dictionary.
    """

    if hasattr(result, "to_dict"):
        return result.to_dict()

    return {
        "step_id": getattr(
            result,
            "step_id",
            None,
        ),
        "tool": getattr(
            result,
            "tool",
            None,
        ),
        "success": getattr(
            result,
            "success",
            False,
        ),
        "output": getattr(
            result,
            "output",
            None,
        ),
        "error": getattr(
            result,
            "error",
            None,
        ),
    }


def _execution_results_to_dict(
    execution_result: Any,
) -> List[Dict[str, Any]]:
    """Convert all executor step results into dictionaries."""

    return [
        _result_to_dict(result)
        for result in execution_result.results
    ]


def _build_execution_error(
    results: List[Dict[str, Any]],
) -> str:
    """Return the most useful error from failed execution results."""

    for result in reversed(results):
        if not result.get("success", False):
            error = result.get("error")

            if isinstance(error, str) and error.strip():
                return error.strip()

    return "Agent execution failed."


def _exception_message(
    exc: Exception,
) -> str:
    """Return a concise error message."""

    message = str(exc).strip()

    if message:
        return message

    return exc.__class__.__name__


# ---------------------------------------------------------------------------
# Background agent worker
# ---------------------------------------------------------------------------


async def _run_agent_task(
    task: AgentTaskState,
    planner: Planner,
    executor: Executor,
    history: List[Dict[str, str]],
    context_chunks: List[Dict[str, Any]],
    file_id: Optional[str],
) -> None:
    """
    Execute one agent task.

    Flow:

        User request
            ↓
        Planner
            ↓
        Plan
            ↓
        Executor
            ↓
        Runtime Memory
            ↓
        Final task status
    """

    try:
        # -------------------------------------------------------------------
        # Ensure runtime memory exists
        # -------------------------------------------------------------------

        if task.memory is None:
            task.memory = get_memory(
                task_id=task.task_id,
                conversation_id=task.conversation_id,
            )

        memory = task.memory

        # -------------------------------------------------------------------
        # Planning
        # -------------------------------------------------------------------

        task.status = TASK_STATUS_PLANNING
        task.current_step = None
        task.current_tool = None

        await _update_task(task)

        # Store the current user request in runtime memory.
        #
        # The request was also loaded from persistent conversation history
        # before planning, but runtime memory represents this task's state.
        memory.add_message(
            role="user",
            content=task.message,
        )

        if file_id:
            memory.set_context(
                "file_id",
                file_id,
            )

        if context_chunks:
            memory.set_context(
                "retrieved_context",
                context_chunks,
            )

        plan = await planner.create_plan(
            message=task.message,
            history=history,
            context_chunks=context_chunks or None,
            has_attachment=bool(file_id),
        )

        task.plan = plan

        task.task_type = plan.get(
            "task_type"
        )

        task.goal = plan.get(
            "goal"
        )

        memory.set_task(
            goal=task.goal,
            task_type=task.task_type,
        )

        await _update_task(task)

        # -------------------------------------------------------------------
        # Execution
        # -------------------------------------------------------------------

        task.status = TASK_STATUS_EXECUTING

        await _update_task(task)

        execution_result = await executor.execute(
            plan
        )

        # -------------------------------------------------------------------
        # Store execution results in runtime memory
        # -------------------------------------------------------------------

        task.results = _execution_results_to_dict(
            execution_result
        )

        for result in execution_result.results:
            task.current_step = result.step_id
            task.current_tool = result.tool

            memory.record_step(
                step_id=result.step_id,
                tool=result.tool,
                success=result.success,
                output=result.output,
                error=result.error,
            )

            task.outputs = memory.get_outputs()

            await _update_task(task)

        # -------------------------------------------------------------------
        # Final status
        # -------------------------------------------------------------------

        task.current_step = None
        task.current_tool = None

        task.outputs = memory.get_outputs()

        if execution_result.success:
            task.status = TASK_STATUS_COMPLETED
            task.error = None

        else:
            task.status = TASK_STATUS_FAILED
            task.error = _build_execution_error(
                task.results
            )

        task.completed_at = _utc_now()

        await _update_task(task)

        logger.info(
            "Agent task '%s' finished with status '%s'.",
            task.task_id,
            task.status,
        )

    except asyncio.CancelledError:
        logger.warning(
            "Agent task '%s' was cancelled.",
            task.task_id,
        )

        task.status = TASK_STATUS_FAILED
        task.current_step = None
        task.current_tool = None
        task.error = "Agent task was cancelled."
        task.completed_at = _utc_now()

        await _update_task(task)

        raise

    except Exception as exc:
        logger.exception(
            "Agent task '%s' failed.",
            task.task_id,
        )

        task.status = TASK_STATUS_FAILED
        task.current_step = None
        task.current_tool = None
        task.error = _exception_message(exc)
        task.completed_at = _utc_now()

        await _update_task(task)


# ---------------------------------------------------------------------------
# POST /api/agent/run
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an agent task",
)
async def run_agent(
    payload: AgentRunRequest,
    db=Depends(get_db),
    planner: Planner = Depends(get_planner),
    executor: Executor = Depends(get_executor),
    retriever: Retriever = Depends(get_retriever),
) -> AgentRunResponse:
    """
    Submit a multi-step agent task.

    The endpoint returns immediately with a task_id.

    The actual planning and execution happen in a background asyncio task.

    Use:
        GET /api/agent/status/{task_id}

    to monitor execution.
    """

    message = payload.message.strip()

    # -----------------------------------------------------------------------
    # Resolve or create conversation
    # -----------------------------------------------------------------------

    conversation_id = payload.conversation_id

    try:
        if conversation_id:
            conversation = await crud.get_conversation(
                db,
                conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Conversation '{conversation_id}' not found."
                    ),
                )

            history = await crud.get_conversation_history(
                db,
                conversation_id,
            )

        else:
            conversation = await crud.create_conversation(
                db
            )

            conversation_id = str(
                conversation.id
            )

            history = []

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to resolve/create conversation "
            "for agent task."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not access conversation storage."
            ),
        )

    # -----------------------------------------------------------------------
    # Retrieve local knowledge context
    # -----------------------------------------------------------------------

    context_chunks: List[Dict[str, Any]] = []

    try:
        context_chunks = await retriever.retrieve_context(
            query=message,
            file_id=payload.file_id,
            top_k=payload.top_k,
        )

    except Exception:
        # Retrieval is useful grounding context, but failure of the
        # optional knowledge layer should not prevent the agent planner
        # from running.
        logger.exception(
            "Knowledge-base retrieval failed for "
            "agent conversation '%s'.",
            conversation_id,
        )

        context_chunks = []

    # -----------------------------------------------------------------------
    # Create runtime task
    # -----------------------------------------------------------------------

    task_id = str(
        uuid.uuid4()
    )

    memory = get_memory(
        task_id=task_id,
        conversation_id=conversation_id,
    )

    memory.add_message(
        role="user",
        content=message,
    )

    if payload.file_id:
        memory.set_context(
            "file_id",
            payload.file_id,
        )

    if context_chunks:
        memory.set_context(
            "retrieved_context",
            context_chunks,
        )

    task = AgentTaskState(
        task_id=task_id,
        conversation_id=conversation_id,
        message=message,
        status=TASK_STATUS_QUEUED,
        memory=memory,
    )

    await _store_task(task)

    # -----------------------------------------------------------------------
    # Persist the user message
    # -----------------------------------------------------------------------
    #
    # This follows the existing database CRUD contract used by
    # routes_chat.py.
    #
    # If persistence fails, do not destroy the agent task itself because
    # runtime execution can still proceed.

    try:
        await crud.add_message(
            db,
            conversation_id,
            role="user",
            content=message,
        )

    except Exception:
        logger.exception(
            "Failed to persist agent user message "
            "for conversation '%s'.",
            conversation_id,
        )

    # -----------------------------------------------------------------------
    # Start background execution
    # -----------------------------------------------------------------------

    asyncio.create_task(
        _run_agent_task(
            task=task,
            planner=planner,
            executor=executor,
            history=history,
            context_chunks=context_chunks,
            file_id=payload.file_id,
        )
    )

    logger.info(
        "Agent task '%s' submitted for conversation '%s'.",
        task_id,
        conversation_id,
    )

    return AgentRunResponse(
        task_id=task_id,
        conversation_id=conversation_id,
        status=TASK_STATUS_QUEUED,
        message=(
            "Agent task accepted. "
            "Use the task_id to check execution status."
        ),
    )


# ---------------------------------------------------------------------------
# GET /api/agent/status/{task_id}
# ---------------------------------------------------------------------------


@router.get(
    "/status/{task_id}",
    response_model=AgentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get agent task status",
)
async def get_agent_status(
    task_id: str,
) -> AgentStatusResponse:
    """
    Return the current runtime state of an agent task.
    """

    task_id = task_id.strip()

    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="task_id must not be empty.",
        )

    task = await _get_task(
        task_id
    )

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Agent task '{task_id}' not found."
            ),
        )

    return AgentStatusResponse(
        task_id=task.task_id,
        conversation_id=task.conversation_id,
        status=task.status,
        message=task.message,
        task_type=task.task_type,
        goal=task.goal,
        current_step=task.current_step,
        current_tool=task.current_tool,
        plan=task.plan,
        results=task.results,
        outputs=task.outputs,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )


__all__ = [
    "router",
    "AgentRunRequest",
    "AgentRunResponse",
    "AgentStatusResponse",
]