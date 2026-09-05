git"""
backend/api/routes_knowledge.py

Knowledge Base API controller.

Endpoints:
    POST /api/knowledge/ingest
    GET  /api/knowledge/search
    GET  /api/knowledge/health
    GET  /api/knowledge/documents
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from backend.api.routes_upload import (
    _file_path as _upload_file_path,
    _find_meta_path as _find_upload_meta_path,
    _read_metadata as _read_upload_metadata,
)
from backend.api.schemas import BaseSchema, SourceReference
from backend.knowledge_base.ingestion import STATUS_SUCCESS, ingest_file
from backend.multimodel.pdf_parser import PDFParser
from backend.knowledge_base.vector_store import get_vector_store
from backend.knowledge_base.retriever import Retriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/knowledge",
    tags=["Knowledge Base"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class KnowledgeSearchResponse(BaseSchema):
    query: str
    results: List[SourceReference]
    count: int


class KnowledgeHealthStatus(BaseSchema):
    status: str
    knowledge_base_ready: bool
    checked_at: datetime


class KnowledgeIngestRequest(BaseSchema):
    source_path: Optional[str] = None
    file_id: Optional[str] = None


class KnowledgeIngestResponse(BaseSchema):
    success: bool
    status: str
    file_id: str
    filename: str
    chunk_count: int
    indexed_count: int = 0
    error: Optional[str] = None
class KnowledgeParseRequest(BaseSchema):
    file_id: str


class KnowledgeParseResponse(BaseSchema):
    success: bool
    status: str
    file_id: str
    filename: str
    page_count: int
    requires_ocr: bool
    ocr_pages: List[int]
    pages: List[dict]
    metadata: dict
    error: Optional[str] = None    


class KnowledgeDocumentsResponse(BaseSchema):
    supported: bool
    documents: List[str]
    message: str


# ---------------------------------------------------------------------------
# POST /api/knowledge/ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=KnowledgeIngestResponse,
)
async def ingest_document(
    payload: KnowledgeIngestRequest,
):
    """
    Extract, normalize, chunk and index a local document.

    source_path:
        Path relative to data/documents/

    file_id:
        Previously uploaded file ID from POST /api/upload
    """

    # Exactly one input is required.
    if not payload.source_path and not payload.file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Provide exactly one of 'source_path' or 'file_id'."
            ),
        )

    if payload.source_path and payload.file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Provide only one of 'source_path' or 'file_id', not both."
            ),
        )
@router.post(
    "/parse",
    response_model=KnowledgeParseResponse,
)
async def parse_uploaded_document(
    payload: KnowledgeParseRequest,
) -> KnowledgeParseResponse:
    """
    Parse a previously uploaded PDF using the existing PDFParser.

    This endpoint does not perform BGE-M3 embedding or ChromaDB indexing.
    """

    meta_path = _find_upload_meta_path(payload.file_id)

    if meta_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No uploaded file found with "
                f"file_id='{payload.file_id}'."
            ),
        )

    try:
        record = _read_upload_metadata(meta_path)
    except Exception:
        logger.exception(
            "Failed to read upload metadata for file_id='%s'.",
            payload.file_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file metadata.",
        )

    target_path = _upload_file_path(
        meta_path.parent.name,
        payload.file_id,
        record.get("extension", ""),
    )

    parser = PDFParser()
    result = parser.parse(target_path)

    return KnowledgeParseResponse(
        success=result.success,
        status=result.status,
        file_id=payload.file_id,
        filename=result.filename,
        page_count=result.page_count,
        requires_ocr=result.requires_ocr,
        ocr_pages=result.ocr_pages,
        pages=[
            {
                "page_number": page.page_number,
                "text": page.text,
                "has_text": page.has_text,
                "requires_ocr": page.requires_ocr,
            }
            for page in result.pages
        ],
        metadata=result.metadata,
        error=result.error,
    )
    # -----------------------------------------------------------------------
    # Resolve uploaded file_id -> actual filesystem path
    # -----------------------------------------------------------------------

    if payload.file_id:

        meta_path = _find_upload_meta_path(payload.file_id)

        if meta_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No uploaded file found with "
                    f"file_id='{payload.file_id}'."
                ),
            )

        try:
            record = _read_upload_metadata(meta_path)
        except Exception:
            logger.exception(
                "Failed to read upload metadata for file_id='%s'.",
                payload.file_id,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to read uploaded file metadata.",
            )

        target_path = _upload_file_path(
            meta_path.parent.name,
            payload.file_id,
            record.get("extension", ""),
        )

    # -----------------------------------------------------------------------
    # Resolve curated document path
    # -----------------------------------------------------------------------

    else:

        documents_root = Path(
            os.getenv(
                "KB_DOCUMENTS_DIR",
                "data/documents",
            )
        )

        target_path = documents_root / payload.source_path

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    result = ingest_file(target_path)

    # -----------------------------------------------------------------------
    # If ingestion failed, do NOT attempt vector indexing.
    # -----------------------------------------------------------------------

    if result.status != STATUS_SUCCESS:

        response = KnowledgeIngestResponse(
            success=result.success,
            status=result.status,
            file_id=result.file_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            indexed_count=0,
            error=result.error,
        )

        status_map = {
            "not_found": status.HTTP_404_NOT_FOUND,
            "unsupported_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "path_denied": status.HTTP_400_BAD_REQUEST,
            "requires_ocr": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "empty": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "malformed": status.HTTP_422_UNPROCESSABLE_ENTITY,
        }

        http_status = status_map.get(
            result.status,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

        return JSONResponse(
            status_code=http_status,
            content=response.model_dump(mode="json"),
        )

    # -----------------------------------------------------------------------
    # Convert chunks -> vector-store documents
    # -----------------------------------------------------------------------

    vector_store = get_vector_store()

    documents = []

    for index, chunk in enumerate(result.chunks):

        documents.append(
            {
                "id": f"{result.file_id}-chunk-{index}",
                "document_id": result.file_id,
                "title": result.filename,
                "text": chunk,
                "file_id": payload.file_id or result.file_id,
                "chunk_index": index,
                "source_path": result.source_path,
            }
        )

    # -----------------------------------------------------------------------
    # BGE-M3 embedding + ChromaDB indexing
    # -----------------------------------------------------------------------

    try:

        indexed_count = await vector_store.add_documents(
            documents
        )

    except FileNotFoundError as exc:

        logger.exception(
            "Embedding model is missing during indexing."
        )

        response = KnowledgeIngestResponse(
            success=False,
            status="embedding_model_missing",
            file_id=result.file_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            indexed_count=0,
            error=str(exc),
        )

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    except Exception as exc:

        logger.exception(
            "Vector-store indexing failed for file_id='%s'.",
            result.file_id,
        )

        response = KnowledgeIngestResponse(
            success=False,
            status="indexing_failed",
            file_id=result.file_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            indexed_count=0,
            error=str(exc),
        )

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=response.model_dump(mode="json"),
        )

    # -----------------------------------------------------------------------
    # Success
    # -----------------------------------------------------------------------

    return KnowledgeIngestResponse(
        success=True,
        status="success",
        file_id=result.file_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        indexed_count=indexed_count,
        error=None,
    )


# ---------------------------------------------------------------------------
# GET /api/knowledge/search
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=KnowledgeSearchResponse,
)
async def search_knowledge_base(
    query: str = Query(
        ...,
        min_length=1,
        description="Search text.",
    ),
    file_id: Optional[str] = Query(
        None,
        description="Restrict search to a specific uploaded file.",
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=50,
        description="Maximum number of results.",
    ),
    retriever: Retriever = Depends(get_retriever),
) -> KnowledgeSearchResponse:

    try:

        raw_results = await retriever.retrieve_context(
            query=query,
            file_id=file_id,
            top_k=top_k,
        )

    except Exception:

        logger.exception(
            "Knowledge base search failed for query=%r.",
            query,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The knowledge base backend failed "
                "to complete the search."
            ),
        )

    results = [
        SourceReference(**item)
        for item in raw_results
    ]

    return KnowledgeSearchResponse(
        query=query,
        results=results,
        count=len(results),
    )


# ---------------------------------------------------------------------------
# GET /api/knowledge/health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=KnowledgeHealthStatus,
)
async def get_knowledge_health(
    retriever: Retriever = Depends(get_retriever),
) -> KnowledgeHealthStatus:

    try:

        ready = await retriever.health_check()

    except Exception:

        logger.warning(
            "Knowledge base health check failed.",
            exc_info=True,
        )

        ready = False

    return KnowledgeHealthStatus(
        status="ok" if ready else "degraded",
        knowledge_base_ready=ready,
        checked_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# GET /api/knowledge/documents
# ---------------------------------------------------------------------------


@router.get(
    "/documents",
    response_model=KnowledgeDocumentsResponse,
)
async def list_knowledge_documents():

    return KnowledgeDocumentsResponse(
        supported=False,
        documents=[],
        message=(
            "Document listing is not currently supported by "
            "the Retriever interface."
        ),
    )