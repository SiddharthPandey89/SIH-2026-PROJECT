"""
database/init_db.py

Async SQLAlchemy 2.x engine/session setup for the Sovereign AI Workbench.

Role of this module:
    - Create the async engine + session factory used by every request.
    - Expose get_db(), the FastAPI dependency backend/api/chat.py (and
      database/crud.py, once implemented) rely on to obtain a session.
    - Expose init_db(), which creates tables from database/db_models.py's
      Base.metadata -- call this once at application startup (e.g. from
      backend/main.py's startup/lifespan handler), not from this module.

Why async SQLAlchemy: backend/api/chat.py awaits every database/crud.py
call (`await crud.get_conversation(db, ...)`, etc.) inside async route
handlers. An AsyncSession is what makes those awaits meaningful -- a plain
sync Session would block the event loop on every query.

Explicitly OUT of scope for this module:
    - Any query/CRUD logic -- that's database/crud.py.
    - Changing database/db_models.py's schema; this module only imports
      Base from it to create tables.

Configuration (environment variables):
    DATABASE_URL   Full SQLAlchemy async URL. Defaults to a local SQLite
                   file at ./data/workbench.db -- fully offline, no server
                   process required. Example for a local/internal Postgres:
                   postgresql+asyncpg://user:pass@localhost:5432/workbench
    DB_ECHO        "true"/"false" (default "false"). Logs all SQL when true;
                   useful for debugging, noisy in production.

Driver note: the default SQLite URL requires the `aiosqlite` package; a
Postgres DATABASE_URL requires `asyncpg`. Add whichever you use to
requirements.txt (not modified by this file).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.database.db_models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_SQLITE_PATH = Path("data") / "workbench.db"
_DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{_DEFAULT_SQLITE_PATH.as_posix()}"

DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)
DB_ECHO: bool = os.getenv("DB_ECHO", "false").strip().lower() in {"1", "true", "yes"}

# Best-effort, defense-in-depth guard mirroring backend/model_router/router.py's
# _assert_local_endpoint(): a configured DATABASE_URL should never point at a
# managed cloud database service. Real enforcement of "nothing leaves the
# premises" is backend/security/network_guard.py at the infra level; this
# just fails fast on an obvious misconfiguration.
_BLOCKED_DATABASE_HOST_SUBSTRINGS = (
    "rds.amazonaws.com",
    "database.azure.com",
    "database.windows.net",
    ".neon.tech",
    ".supabase.co",
    "cloud.google.com",
)


class DatabaseConfigError(Exception):
    """Raised when DATABASE_URL is invalid or points at a disallowed external host."""


def _assert_local_database_url(url: str) -> None:
    lowered = url.lower()
    for blocked in _BLOCKED_DATABASE_HOST_SUBSTRINGS:
        if blocked in lowered:
            raise DatabaseConfigError(
                f"DATABASE_URL appears to point at a managed cloud database ('{blocked}'). "
                "This workbench is local/offline-first only -- point DATABASE_URL at a local "
                "SQLite file or a database on your own internal network instead."
            )


def _mask_credentials(url: str) -> str:
    """Redact a password in a DB URL before logging it, e.g. postgresql+asyncpg://user:***@host/db."""
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", url)


_assert_local_database_url(DATABASE_URL)

if DATABASE_URL == _DEFAULT_DATABASE_URL:
    # Ensure the local SQLite file's parent directory exists so the very
    # first connection doesn't fail with "unable to open database file".
    _DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

logger.info("Database configured: %s (echo=%s)", _mask_credentials(DATABASE_URL), DB_ECHO)


# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=DB_ECHO, future=True)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """
    Create all tables defined on Base.metadata (from database/db_models.py)
    if they don't already exist. Call once at application startup -- e.g.
    from backend/main.py's FastAPI startup/lifespan handler.

    Safe to call multiple times: SQLAlchemy's create_all() is a no-op for
    tables that already exist.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ensured (create_all against %s).", _mask_credentials(DATABASE_URL))
    except Exception:
        logger.exception("Failed to initialize database schema.")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding a request-scoped AsyncSession.

    Lifecycle:
        - A fresh session is opened per request.
        - If an unhandled exception propagates out of the route while this
          session is active, it is rolled back so no partial write lingers.
        - The session is always closed at the end of the request, whether
          it succeeded or failed.

    Commit responsibility is intentionally left to database/crud.py, not
    this dependency: crud functions commit after the specific write(s) they
    perform, since a single request (e.g. POST /api/chat) may need to
    persist more than one independent write -- a user message, then an
    assistant message -- within the same session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            logger.exception("Unhandled error during a database session; rolling back.")
            await session.rollback()
            raise
        finally:
            await session.close()


async def dispose_engine() -> None:
    """
    Cleanly dispose of the engine's connection pool. Call from an
    application shutdown handler for an orderly close; not required for
    correctness (the OS reclaims connections on process exit), but good
    practice.
    """
    await engine.dispose()
    logger.info("Database engine disposed.")
