"""
knowledge_base/retriever.py

Retrieval layer for RAG over the organization's local knowledge base:
data/documents/manuals/, data/documents/sops/, data/documents/past_reports/,
and uploaded documents under data/uploads/.

Role of this module:
    - Expose Retriever.retrieve_context(query, file_id, top_k), returning
      results shaped exactly as backend/api/chat.py expects:
      {"document_id", "title", "snippet", "score"}.
    - Expose Retriever.health_check() and get_retriever(), matching the
      dependency-factory pattern already used by model_router.
    - Depend on knowledge_base/vector_store.py only through an abstract
      VectorStoreBackend contract -- never a concrete import -- so a real
      FAISS/Chroma/Qdrant-backed implementation can be dropped in later
      with zero changes here.

Current state of knowledge_base/vector_store.py and knowledge_base/ingestion.py:
    Neither exists yet. Rather than hard-depend on modules that aren't
    there (or invent an incompatible API for them), this file:
        1. Tries to import a real backend from knowledge_base.vector_store
           at construction time.
        2. Falls back to _LocalFilesystemSearch -- a small, dependency-free
           keyword search over plain-text documents already on disk -- so
           retrieval is genuinely functional today, not just stubbed out.
    Once vector_store.py exists and exposes get_vector_store() returning
    something satisfying VectorStoreBackend, Retriever will pick it up
    automatically; no code in this file needs to change.

Explicitly OUT of scope for this module:
    - Embeddings, chunking-for-ingestion, or vector index construction --
      that's knowledge_base/ingestion.py's and knowledge_base/vector_store.py's
      job once implemented.
    - OCR / PDF parsing of scanned documents -- that's multimodal/. The
      local fallback here only reads plain-text-ish files (.txt, .md);
      scanned PDFs and images require that pipeline to run first.

Fully local/offline: no network calls, no cloud APIs, no external LLM
calls anywhere in this file.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["Retriever", "get_retriever", "VectorStoreBackend"]


# ---------------------------------------------------------------------------
# Extension point: vector store backend contract
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreBackend(Protocol):
    """
    Contract knowledge_base/vector_store.py should implement. Retriever
    depends only on this shape, never on a concrete class -- swapping in
    FAISS/Chroma/Qdrant later requires no changes to this file.
    """

    async def search(self, query: str, top_k: int, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return up to top_k hits, each a dict with document_id/title/snippet/score."""
        ...

    async def is_ready(self) -> bool:
        """Return True if the backend is initialized and can currently serve queries."""
        ...


def _load_vector_store_backend() -> Optional[VectorStoreBackend]:
    """
    Best-effort, optional import of knowledge_base/vector_store.py.

    Returns None (never raises) if that module doesn't exist yet, doesn't
    expose get_vector_store(), or returns something that doesn't look like
    a VectorStoreBackend -- Retriever falls back to local search in every
    one of those cases rather than failing to construct.
    """
    try:
        from backend.knowledge_base.vector_store import get_vector_store  # type: ignore[import-not-found]
    except ImportError:
        logger.info("knowledge_base.vector_store not implemented yet; using local fallback search.")
        return None

    try:
        backend = get_vector_store()
    except Exception:
        logger.exception("knowledge_base.vector_store.get_vector_store() raised; using local fallback search.")
        return None

    if not isinstance(backend, VectorStoreBackend):
        logger.warning(
            "get_vector_store() did not return an object matching VectorStoreBackend; "
            "using local fallback search instead."
        )
        return None

    logger.info("Using vector store backend from knowledge_base.vector_store.")
    return backend


# ---------------------------------------------------------------------------
# Local fallback search (used until vector_store.py / ingestion.py exist)
# ---------------------------------------------------------------------------

_DOCUMENTS_DIR = Path(os.getenv("KB_DOCUMENTS_DIR", "data/documents"))
_UPLOADS_DIR = Path(os.getenv("KB_UPLOADS_DIR", "data/uploads"))
_CURATED_SUBDIRS: Tuple[str, ...] = ("manuals", "sops", "past_reports")

# Intentionally narrow: this fallback only understands plain-text content.
# Scanned PDFs, drawings, and photos need OCR/vision (multimodal/) plus a
# real ingestion pipeline (knowledge_base/ingestion.py) before they're
# searchable text -- out of scope for this file.
_SEARCHABLE_EXTENSIONS = {".txt", ".md"}
_SNIPPET_MAX_CHARS = 400

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_PATTERN.findall(text)]


@dataclass(frozen=True)
class _Chunk:
    document_id: str
    title: str
    text: str


def _iter_source_files(file_id: Optional[str]) -> List[Path]:
    """
    Enumerate candidate local files to search.

    If file_id is given, search is restricted to files under data/uploads
    whose filename contains file_id -- a best-effort match used until
    knowledge_base/ingestion.py can provide a proper file_id -> path
    resolver. Otherwise, search the curated manuals/sops/past_reports roots.
    """
    if file_id:
        if not _UPLOADS_DIR.exists():
            return []
        return [
            path
            for path in _UPLOADS_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in _SEARCHABLE_EXTENSIONS and file_id in path.name
        ]

    files: List[Path] = []
    for subdir in _CURATED_SUBDIRS:
        root = _DOCUMENTS_DIR / subdir
        if root.exists():
            files.extend(
                path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _SEARCHABLE_EXTENSIONS
            )
    return files


def _chunk_file(path: Path) -> List[_Chunk]:
    """Split a file into paragraph-sized chunks for scoring."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        logger.warning("Could not read file '%s' for local search fallback.", path, exc_info=True)
        return []

    document_id = str(path)
    title = path.stem.replace("_", " ").replace("-", " ").title()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    return [_Chunk(document_id=document_id, title=title, text=paragraph) for paragraph in paragraphs]


def _score_chunk(query_terms: List[str], chunk: _Chunk) -> float:
    """Fraction of query terms present in the chunk -- simple, deterministic, no ML model required."""
    if not query_terms:
        return 0.0
    chunk_terms = set(_tokenize(chunk.text))
    matches = sum(1 for term in query_terms if term in chunk_terms)
    return matches / len(query_terms)


class _LocalFilesystemSearch:
    """
    Deterministic, dependency-free keyword search over local plain-text
    documents. Implements the same shape as VectorStoreBackend so Retriever
    can use it interchangeably: no embeddings, no external services, no
    setup required -- just files already sitting under data/documents and
    data/uploads.
    """

    async def search(self, query: str, top_k: int, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_terms = _tokenize(query)
        files = _iter_source_files(file_id)
        if not files or not query_terms:
            return []

        scored: List[Tuple[float, _Chunk]] = []
        for path in files:
            for chunk in _chunk_file(path):
                score = _score_chunk(query_terms, chunk)
                if score > 0.0:
                    scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "snippet": chunk.text[:_SNIPPET_MAX_CHARS],
                "score": round(score, 4),
            }
            for score, chunk in scored[:top_k]
        ]

    async def is_ready(self) -> bool:
        return _DOCUMENTS_DIR.exists() or _UPLOADS_DIR.exists()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Retrieves grounding context for a chat turn.

    Prefers a real vector store backend (knowledge_base/vector_store.py)
    when available; otherwise falls back to local keyword search over
    data/documents and data/uploads. Both paths are fully local/offline.
    """

    def __init__(self, backend: Optional[VectorStoreBackend] = None) -> None:
        self._backend: VectorStoreBackend = backend or _load_vector_store_backend() or _LocalFilesystemSearch()
        logger.info("Retriever initialized with backend: %s", type(self._backend).__name__)

    async def retrieve_context(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return up to top_k relevant chunks for query, each shaped as
        {"document_id": str, "title": str, "snippet": str, "score": float}
        -- ready for backend/api/chat.py to wrap in SourceReference and for
        backend/model_router/router.py to fold into its prompt.

        Returns an empty list (never raises for "nothing found") if the
        query is empty, top_k is non-positive, or no relevant content
        exists. Backend errors (e.g. an unreachable vector DB) do
        propagate -- chat.py already catches retrieval failures and
        degrades gracefully rather than failing the whole request.
        """
        if not isinstance(query, str) or not query.strip():
            logger.debug("retrieve_context called with an empty query; returning no results.")
            return []
        if top_k <= 0:
            return []

        try:
            results = await self._backend.search(query=query.strip(), top_k=top_k, file_id=file_id)
        except Exception:
            logger.exception("Retrieval backend raised while searching for query=%r.", query)
            raise

        return [self._normalize(result) for result in results]

    @staticmethod
    def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
        """Guarantee the exact key set/types backend/api/schemas.SourceReference expects."""
        return {
            "document_id": str(result.get("document_id", "unknown")),
            "title": str(result.get("title", "Untitled")),
            "snippet": str(result.get("snippet", "")),
            "score": float(result.get("score", 0.0)),
        }

    async def health_check(self) -> bool:
        """
        Return True if the retriever can currently serve queries: either
        the real vector store backend reports ready, or (in fallback mode)
        the local document directories exist and are reachable.
        """
        try:
            return await self._backend.is_ready()
        except Exception:
            logger.warning("Retriever health check failed.", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Module-level singleton + FastAPI-style dependency factory
# ---------------------------------------------------------------------------

_default_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Dependency factory returning a shared Retriever instance, matching get_model_router()'s pattern."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever
