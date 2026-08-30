"""
backend/api/routes_knowledge.py

Knowledge Base API controller.

Endpoints:
    POST /api/knowledge/ingest      ingest/index an existing local document or uploaded file
    GET  /api/knowledge/search      search the knowledge base
    GET  /api/knowledge/health      report knowledge-base readiness
    GET  /api/knowledge/documents   list indexed/available documents

Architecture notes (read before changing anything here):

  - Inspected before writing this file: backend/api/routes_chat.py,
    backend/api/routes_upload.py, backend/api/schemas.py,
    knowledge_base/retriever.py, database/ (db_models.py, crud.py,
    init_db.py), and the data/ directory tree.

  - knowledge_base/vector_store.py does NOT exist. knowledge_base/retriever.py
    already accounts for this: it defines a VectorStoreBackend Protocol and
    falls back to an internal, private local keyword search
    (_LocalFilesystemSearch) when no real vector store is registered. This
    file reuses Retriever exactly as-is (retrieve_context / health_check)
    and does not reach into retriever.py's private internals or duplicate
    its search logic.

  - knowledge_base/ingestion.py now exists and implements extraction +
    normalization + deterministic chunking (.txt/.md/.pdf/.docx). POST
    /api/knowledge/ingest calls it directly. This still does NOT build a
    vector index -- knowledge_base/vector_store.py does not exist, so the
    response never claims anything is now searchable; it only reports the
    extraction/chunking outcome (status, chunk_count, etc.).

  - For file_id-based ingestion, this file reuses
    backend/api/routes_upload.py's own (private) file_id -> path lookup
    helpers (_find_meta_path, _read_metadata, _file_path) instead of
    re-implementing the uploads storage scheme here. routes_upload.py
    itself is not modified.

  - For the same reason, Retriever exposes no public "list indexed
    documents" method. GET /api/knowledge/documents does not scan
    data/documents/ or data/uploads/ itself (that would duplicate/guess at
    retriever internals) -- it reports supported=false with an empty list
    and a clear explanation, which is a normal 200 response a frontend can
    render directly ("no listing capability yet"), not an error state.

  - No new database table is created; this controller does not touch
    database/ at all. No cloud APIs or external network calls anywhere --
    knowledge_base.retriever.Retriever is local/offline by construction.

  - Reuses backend/api/schemas.SourceReference and BaseSchema as-is;
    only adds the minimal new schemas this controller's own request/
    response shapes actually need.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from backend.api.routes_upload import _file_path as _upload_file_path
from backend.api.routes_upload import _find_meta_path as _find_upload_meta_path
from backend.api.routes_upload import _read_metadata as _read_upload_metadata
from backend.api.schemas import BaseSchema, SourceReference
from backend.knowledge_base.ingestion import STATUS_SUCCESS, ingest_file
from backend.knowledge_base.retriever import Retriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])


# ---------------------------------------------------------------------------
# Minimal endpoint-specific schemas
# ---------------------------------------------------------------------------


class KnowledgeSearchResponse(BaseSchema):
    query: str
    results: List[SourceReference]
    count: int


class KnowledgeHealthStatus(BaseSchema):
    status: str  # "ok" | "degraded"
    knowledge_base_ready: bool
    checked_at: datetime


class KnowledgeIngestRequest(BaseSchema):
    source_path: Optional[str] = None  # path relative to data/documents/, for a curated local document
    file_id: Optional[str] = None  # id of a file previously stored via POST /api/upload


class KnowledgeIngestResponse(BaseSchema):
    """
    Reports extraction + normalization + chunking only. Never implies a
    vector index was updated -- knowledge_base/vector_store.py doesn't
    exist yet, so nothing here is searchable as a result of this call.
    """

    success: bool
    status: str  # see knowledge_base.ingestion's STATUS_* constants
    file_id: str
    filename: str
    chunk_count: int
    error: Optional[str] = None


class KnowledgeDocumentsResponse(BaseSchema):
    supported: bool
    documents: List[str]
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest_document(payload: KnowledgeIngestRequest):
    """
    Ingest (extract + normalize + chunk) a local document or an uploaded
    file via knowledge_base/ingestion.py.

    This does NOT build a vector index -- knowledge_base/vector_store.py
    does not exist yet, so nothing becomes searchable as a side effect of
    calling this. The response only reports the extraction/chunking
    outcome (status, chunk_count, error), never that indexing happened.
    """
    if not payload.source_path and not payload.file_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Provide exactly one of 'source_path' (a document under data/documents/) or 'file_id' (an uploaded file).",
        )
    if payload.source_path and payload.file_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Provide only one of 'source_path' or 'file_id', not both.",
        )

    if payload.file_id:
        # Reuses routes_upload.py's own file_id -> path lookup rather than
        # re-implementing the uploads storage scheme here.
        meta_path = _find_upload_meta_path(payload.file_id)
        if meta_path is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No uploaded file found with file_id='{payload.file_id}'.")
        try:
            record = _read_upload_metadata(meta_path)
        except Exception:
            logger.exception("Failed to read upload metadata for file_id='%s'.", payload.file_id)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to read uploaded file metadata.")
        target_path = _upload_file_path(meta_path.parent.name, payload.file_id, record.get("extension", ""))
    else:
        documents_root = Path(os.getenv("KB_DOCUMENTS_DIR", "data/documents"))
        target_path = documents_root / payload.source_path

    result = ingest_file(target_path)

    response = KnowledgeIngestResponse(
        success=result.success,
        status=result.status,
        file_id=result.file_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        error=result.error,
    )

    if result.status == STATUS_SUCCESS:
        return response

    status_map = {
        "not_found": status.HTTP_404_NOT_FOUND,
        "unsupported_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "path_denied": status.HTTP_400_BAD_REQUEST,
        "requires_ocr": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "empty": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "malformed": status.HTTP_422_UNPROCESSABLE_ENTITY,
    }
    http_status = status_map.get(result.status, status.HTTP_422_UNPROCESSABLE_ENTITY)
    return JSONResponse(status_code=http_status, content=response.model_dump(mode="json"))


@router.get("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge_base(
    query: str = Query(..., min_length=1, description="Search text."),
    file_id: Optional[str] = Query(None, description="Restrict search to a specific uploaded file, if supported."),
    top_k: int = Query(5, ge=1, le=50, description="Maximum number of results to return."),
    retriever: Retriever = Depends(get_retriever),
) -> KnowledgeSearchResponse:
    """
    Search the knowledge base via the existing Retriever
    (knowledge_base/retriever.py) -- the same component
    backend/api/routes_chat.py already uses for RAG grounding.
    """
    try:
        raw_results = await retriever.retrieve_context(query=query, file_id=file_id, top_k=top_k)
    except Exception:
        logger.exception("Knowledge base search failed for query=%r.", query)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The knowledge base backend failed to complete the search.")

    results = [SourceReference(**item) for item in raw_results]
    return KnowledgeSearchResponse(query=query, results=results, count=len(results))


@router.get("/health", response_model=KnowledgeHealthStatus)
async def get_knowledge_health(retriever: Retriever = Depends(get_retriever)) -> KnowledgeHealthStatus:
    """Report whether the knowledge base (Retriever and its backend) is currently usable."""
    try:
        ready = await retriever.health_check()
    except Exception:
        logger.warning("Knowledge base health check failed.", exc_info=True)
        ready = False

    return KnowledgeHealthStatus(
        status="ok" if ready else "degraded",
        knowledge_base_ready=ready,
        checked_at=datetime.utcnow(),
    )


@router.get("/documents", response_model=KnowledgeDocumentsResponse)
async def list_knowledge_documents() -> KnowledgeDocumentsResponse:
    """
    List indexed/available documents, if the current knowledge-base
    implementation supports it.

    It does not: Retriever exposes no public method for enumerating what
    it can search over (only retrieve_context() and health_check()), and
    its local fallback's file discovery is a private implementation
    detail this controller does not reach into. Returns a normal 200 with
    supported=false rather than guessing at document names or an error.
    """
    return KnowledgeDocumentsResponse(
        supported=False,
        documents=[],
        message=(
            "Document listing is not supported by the current knowledge_base.retriever.Retriever "
            "implementation -- it exposes search() and health_check() only. Add a document-listing "
            "method to the VectorStoreBackend contract (and knowledge_base/vector_store.py) to enable this."
        ),
    )