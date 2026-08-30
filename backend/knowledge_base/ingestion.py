"""
knowledge_base/ingestion.py

Local document ingestion pipeline for the Sovereign AI Workbench.

Pipeline (exactly this, nothing more):
    file -> validate path/type -> extract text -> normalize -> deterministic
    chunking -> IngestionResult

This module does NOT build a vector index and does NOT claim documents are
now searchable. knowledge_base/vector_store.py does not exist yet, and this
file deliberately does not create it. Once a real vector store exists,
something else (e.g. routes_knowledge.py) can call ingest_file() and then
hand its chunks to that store -- ingestion and indexing are kept separate.

Explicitly OUT of scope for this module:
    - OCR of scanned/image-only PDFs. If a PDF has no extractable text
      layer, ingest_file() returns status="requires_ocr" and stops --
      the multimodal/ OCR pipeline (not yet implemented) owns that case.
    - Vector storage/indexing -- see above.
    - Search/retrieval -- that remains entirely knowledge_base/retriever.py's
      job. This module is never imported by retriever.py and does not
      duplicate any of its logic.
    - Resolving an upload's file_id to a filesystem path -- that scheme is
      owned by backend/api/routes_upload.py. This module only accepts an
      already-resolved path and validates it against the two allowed
      roots; the caller (routes_knowledge.py) is responsible for turning a
      file_id into a path using routes_upload.py's own lookup.

Security:
    - Every path is resolved (symlinks included) and must land inside
      data/documents/ or data/uploads/ (configurable via the same
      KB_DOCUMENTS_DIR / KB_UPLOADS_DIR env vars knowledge_base/retriever.py
      already uses, so both modules always agree on the allowed roots).
      Anything outside -> status="path_denied", nothing is read.
    - Only a fixed extension allowlist is processed; everything else is
      rejected before the file is opened.
    - No shell commands, no subprocess calls, no code execution of any
      kind against file contents -- only pure-Python text extraction
      (pypdf, python-docx) and stdlib file reads.
    - No network or cloud calls anywhere in this file.

Error handling:
    - ingest_file() never raises for an expected failure (missing file,
      wrong type, path outside the allowed roots, empty content, a
      corrupt/malformed document, a scanned PDF needing OCR). Every one of
      those is reported as a structured, non-2xx-safe IngestionResult so a
      FastAPI route calling this can build a normal HTTP response without
      a try/except around business-as-usual failures.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
import pypdf
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pypdf
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

__all__ = ["IngestionResult", "ingest_file", "DEFAULT_CHUNK_SIZE", "DEFAULT_CHUNK_OVERLAP", "SUPPORTED_EXTENSIONS"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Same env vars as knowledge_base/retriever.py, so ingestion and retrieval
# never disagree about where the knowledge base's files live.

_DOCUMENTS_ROOT = Path(os.getenv("KB_DOCUMENTS_DIR", "data/documents")).resolve()
_UPLOADS_ROOT = Path(os.getenv("KB_UPLOADS_DIR", "data/uploads")).resolve()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 150  # characters

# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_NOT_FOUND = "not_found"
STATUS_UNSUPPORTED_TYPE = "unsupported_type"
STATUS_PATH_DENIED = "path_denied"
STATUS_EMPTY = "empty"
STATUS_REQUIRES_OCR = "requires_ocr"
STATUS_MALFORMED = "malformed"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Structured outcome of one ingest_file() call. Always returned, never an exception."""

    success: bool
    status: str
    file_id: str
    source_path: str
    filename: str
    text: str
    chunks: List[str] = field(default_factory=list)
    chunk_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------


def _is_within_allowed_roots(resolved: Path) -> bool:
    return resolved.is_relative_to(_DOCUMENTS_ROOT) or resolved.is_relative_to(_UPLOADS_ROOT)


def _derive_file_id(resolved: Path) -> str:
    """Deterministic id from the resolved path -- same file always yields the same id, no random state."""
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Text extraction (one function per supported type)
# ---------------------------------------------------------------------------


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_md(path: Path) -> str:
    # Markdown is ingested as plain text -- headings/lists stay as literal
    # characters, which is fine for chunking/search purposes at this stage.
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(path: Path) -> Tuple[str, bool]:
    """
    Returns (text, requires_ocr). requires_ocr is True when the PDF has
    pages but none of them yield any extractable text -- the signature of
    a scanned/image-only PDF that pypdf cannot read text from.
    """
    reader = pypdf.PdfReader(str(path))
    if reader.is_encrypted:
        # Try an empty password (common for "protected but not really
        # secret" PDFs); if that fails, treat it as malformed rather than
        # guessing further.
        if reader.decrypt("") == pypdf.PasswordType.NOT_DECRYPTED:
            raise ValueError("PDF is password-protected and could not be opened.")

    page_texts = [(page.extract_text() or "") for page in reader.pages]
    combined = "\n\n".join(page_texts).strip()
    requires_ocr = len(combined) == 0 and len(reader.pages) > 0
    return combined, requires_ocr


def extract_docx(path: Path) -> str:
    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs).strip()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Unify line endings/unicode form and collapse noisy whitespace, without altering wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Deterministic chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Fixed-size, character-based sliding-window chunking with overlap.
    Deterministic: the same text and parameters always produce the same
    chunks. No ML tokenizer or external dependency required.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size.")
    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == text_length:
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest_file(
    path: Union[str, Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestionResult:
    """
    Validate, extract, normalize, and chunk one local file.

    path must resolve to somewhere inside data/documents/ or data/uploads/
    (or their KB_DOCUMENTS_DIR / KB_UPLOADS_DIR overrides) -- anything else
    is rejected without being opened. Never raises: every failure mode
    (missing file, disallowed path, unsupported type, empty/malformed
    content, a scanned PDF needing OCR) comes back as a non-success
    IngestionResult instead of an exception.
    """
    raw_path = Path(path)
    filename = raw_path.name

    def _fail(status_: str, error: str, resolved_str: str = "") -> IngestionResult:
        logger.warning("Ingestion failed for '%s': [%s] %s", raw_path, status_, error)
        return IngestionResult(
            success=False,
            status=status_,
            file_id="",
            source_path=resolved_str or str(raw_path),
            filename=filename,
            text="",
            chunks=[],
            chunk_count=0,
            error=error,
        )

    try:
        resolved = raw_path.resolve()
    except OSError as exc:
        return _fail(STATUS_PATH_DENIED, f"Could not resolve path: {exc}")

    if not _is_within_allowed_roots(resolved):
        return _fail(
            STATUS_PATH_DENIED,
            "Path is outside the allowed data/documents/ or data/uploads/ directories.",
            str(resolved),
        )

    extension = resolved.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return _fail(STATUS_UNSUPPORTED_TYPE, f"Unsupported file type '{extension or 'unknown'}'.", str(resolved))

    if not resolved.is_file():
        return _fail(STATUS_NOT_FOUND, "File does not exist.", str(resolved))

    requires_ocr = False
    try:
        if extension == ".txt":
            raw_text = extract_txt(resolved)
        elif extension == ".md":
            raw_text = extract_md(resolved)
        elif extension == ".pdf":
            raw_text, requires_ocr = extract_pdf(resolved)
        elif extension == ".docx":
            raw_text = extract_docx(resolved)
        else:  # unreachable given the extension check above; kept as a safety net
            return _fail(STATUS_UNSUPPORTED_TYPE, f"Unsupported file type '{extension}'.", str(resolved))
    except Exception as exc:  # noqa: BLE001 - any extractor failure is reported, never propagated
        return _fail(STATUS_MALFORMED, f"Failed to extract text: {exc}", str(resolved))

    if requires_ocr:
        logger.info("'%s' appears to be a scanned/image-only PDF; deferring to the OCR pipeline.", resolved)
        return IngestionResult(
            success=False,
            status=STATUS_REQUIRES_OCR,
            file_id=_derive_file_id(resolved),
            source_path=str(resolved),
            filename=filename,
            text="",
            chunks=[],
            chunk_count=0,
            error="No extractable text layer; route this file through the multimodal OCR pipeline instead.",
        )

    normalized = normalize_text(raw_text)
    if not normalized:
        return _fail(STATUS_EMPTY, "No text content found in file.", str(resolved))

    try:
        chunks = chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError as exc:
        return _fail(STATUS_MALFORMED, f"Invalid chunking parameters: {exc}", str(resolved))

    file_id = _derive_file_id(resolved)
    logger.info("Ingested '%s' -> %d chunk(s) (file_id=%s).", resolved, len(chunks), file_id)
    return IngestionResult(
        success=True,
        status=STATUS_SUCCESS,
        file_id=file_id,
        source_path=str(resolved),
        filename=filename,
        text=normalized,
        chunks=chunks,
        chunk_count=len(chunks),
        error=None,
    )