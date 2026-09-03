"""
backend/api/routes_chat.py

Chat API controller: POST /api/chat, GET /api/chat/health.
Routing/validation/error-handling only -- no LLM, RAG, OCR, SQL, or agent
logic here. Expects schemas.py, model_router/, knowledge_base/retriever.py,
database/crud.py and database/init_db.py to be implemented separately.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.schemas import ChatRequest, ChatResponse, ChatHealthStatus, SourceReference
from backend.model_router.router import ModelRouter, get_model_router
from backend.model_router.task_classifier import classify_task
from backend.knowledge_base.retriever import Retriever, get_retriever
from backend.database import crud
from backend.database.init_db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])

RAG_TASK_TYPES = {"document_qa", "summarization"}


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def post_chat(
    payload: ChatRequest,
    db=Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router),
    retriever: Retriever = Depends(get_retriever),
) -> ChatResponse:
    message = payload.message.strip()
    if not message:
    
        raise HTTPException(
            status_code=400,
            detail="`message` must not be empty.",
        )    
    # Resolve or create the conversation
    conversation_id = payload.conversation_id
    try:
        if conversation_id:
            conversation = await crud.get_conversation(db, conversation_id)
            if conversation is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"Conversation '{conversation_id}' not found.")
            history = await crud.get_conversation_history(db, conversation_id)
        else:
            conversation = await crud.create_conversation(db)
            conversation_id = str(conversation.id)
            history = []
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to resolve/create conversation.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not access conversation storage.")

    # Classify task type
    try:
        task_type = await classify_task(message, has_attachment=bool(payload.file_id))
    except Exception:
        logger.exception("Task classification failed; falling back to 'chat'.")
        task_type = "chat"

    # Retrieve grounding context when relevant
    source_chunks: List[Dict[str, Any]] = []
    if payload.file_id or task_type in RAG_TASK_TYPES:
        try:
            source_chunks = await retriever.retrieve_context(query=message, file_id=payload.file_id, top_k=5)
        except Exception:
            logger.exception("Knowledge base retrieval failed for conversation %s.", conversation_id)
            source_chunks = []

    # Generate via local model router
    try:
        generation = await model_router.generate(
            message=message,
            task_type=task_type,
            history=history,
            context_chunks=source_chunks or None,
        )
        answer = generation["answer"]
        model_used = generation["model"]
    except Exception:
        logger.exception("Model generation failed for conversation %s (task_type=%s).", conversation_id, task_type)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The local model backend failed to generate a response.")

    # Persist turn
    try:
        await crud.add_message(db, conversation_id, role="user", content=message)
        await crud.add_message(
            db, conversation_id, role="assistant", content=answer,
            metadata={"model": model_used, "task_type": task_type},
        )
    except Exception:
        logger.exception("Failed to persist conversation history for conversation %s.", conversation_id)

    sources = [
        SourceReference(
            document_id=c.get("document_id", "unknown"),
            title=c.get("title", "Untitled"),
            snippet=c.get("snippet", ""),
            score=float(c.get("score", 0.0)),
        )
        for c in source_chunks
    ]

    return ChatResponse(
        answer=answer,
        model=model_used,
        task_type=task_type,
        conversation_id=conversation_id,
        sources=sources,
        created_at=datetime.utcnow(),
    )


@router.get("/health", response_model=ChatHealthStatus, status_code=status.HTTP_200_OK)
async def get_chat_health(
    db=Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router),
    retriever: Retriever = Depends(get_retriever),
) -> ChatHealthStatus:
    model_router_ready = False
    knowledge_base_ready = False
    database_ready = False

    try:
        model_router_ready = await model_router.health_check()
    except Exception:
        logger.warning("Model router health check failed.", exc_info=True)

    try:
        knowledge_base_ready = await retriever.health_check()
    except Exception:
        logger.warning("Knowledge base health check failed.", exc_info=True)

    try:
        database_ready = await crud.ping(db)
    except Exception:
        logger.warning("Database health check failed.", exc_info=True)

    overall_status = "ok" if (model_router_ready and knowledge_base_ready and database_ready) else "degraded"

    return ChatHealthStatus(
        status=overall_status,
        model_router_ready=model_router_ready,
        knowledge_base_ready=knowledge_base_ready,
        database_ready=database_ready,
        checked_at=datetime.utcnow(),
    )