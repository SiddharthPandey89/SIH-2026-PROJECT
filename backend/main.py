"""
backend/main.py

FastAPI application entrypoint for the Sovereign AI Workbench.

Role of this module:
    - Construct the FastAPI app with proper metadata.
    - Manage application lifecycle: initialize the local database on
      startup, cleanly dispose its engine on shutdown.
    - Register backend/api/routes_chat.py's router.
    - Expose a minimal "/" info endpoint and a basic "/health" liveness
      endpoint.

Explicitly OUT of scope for this module:
    - Any LLM, RAG, agent, or OCR logic -- those live in their own modules
      and are only reached through the routers this file registers.
    - Deep dependency health checks (model router / knowledge base /
      database readiness) -- that's GET /api/chat/health, already
      implemented in backend/api/routes_chat.py. GET /health here is a
      lightweight "is the process up" liveness probe, not a readiness
      check for those dependencies.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import FastAPI

from backend.api.routes_chat import router as chat_router
from backend.api.routes_upload import router as upload_router
from backend.database.init_db import dispose_engine, init_db
from backend.api.routes_knowledge import router as knowledge_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_NAME = "Sovereign AI Workbench"
APP_DESCRIPTION = (
    "Self-hosted, air-gapped agentic AI workbench for confidential industrial work. "
    "Runs entirely on local, open-weight models -- no cloud APIs, no external network calls."
)
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifecycle.

    Startup:  create database tables (via database/init_db.py's init_db())
              if they don't already exist.
    Shutdown: cleanly dispose the database engine's connection pool (via
              database/init_db.py's dispose_engine()).
    """
    logger.info("Starting %s v%s ...", APP_NAME, APP_VERSION)
    await init_db()
    logger.info("Database initialized.")
    try:
        yield
    finally:
        logger.info("Shutting down %s ...", APP_NAME)
        await dispose_engine()
        logger.info("Database engine disposed.")


app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(knowledge_router)


@app.get("/", tags=["Meta"], summary="Basic service info")
async def root() -> Dict[str, str]:
    """Minimal landing endpoint confirming the service is up and identifying itself."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["Meta"], summary="Basic liveness check")
async def health() -> Dict[str, str]:
    """
    Lightweight liveness probe: confirms the process is running and able to
    respond at all. For a readiness check of the chat pipeline's actual
    dependencies (model router, knowledge base, database), see
    GET /api/chat/health instead.
    """
    return {"status": "ok"}