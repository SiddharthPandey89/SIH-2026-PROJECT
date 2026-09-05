"""
backend/multimodal/pdf_parser.py

PDF parsing and modality detection for the Sovereign AI Workbench.

Responsibilities:
    - Validate and open a local PDF.
    - Extract page-level text using pypdf.
    - Detect whether each page has an extractable text layer.
    - Detect scanned/image-only PDFs.
    - Preserve page-level information required by the OCR pipeline.
    - Provide structured parser results for downstream multimodal processing.

This module does NOT:
    - Perform OCR.
    - Generate embeddings.
    - Build/search the vector store.
    - Execute arbitrary code.
    - Make network/cloud calls.
    - Replace knowledge_base.ingestion.py.

For scanned pages/PDFs, this module reports that OCR is required.
The actual OCR work belongs to backend/multimodal/ocr_pipeline.py.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pypdf

logger = logging.getLogger(__name__)

__all__ = [
    "PDFPage",
    "PDFParseResult",
    "PDFParser",
    "parse_pdf",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DOCUMENTS_ROOT = Path(
    os.getenv("KB_DOCUMENTS_DIR", "data/documents")
).resolve()

_UPLOADS_ROOT = Path(
    os.getenv("KB_UPLOADS_DIR", "data/uploads")
).resolve()

MIN_TEXT_CHARS = 20


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class PDFPage:
    """
    Information extracted from one PDF page.
    """

    page_number: int
    text: str
    has_text: bool
    requires_ocr: bool


@dataclass
class PDFParseResult:
    """
    Structured result returned by the PDF parser.
    """

    success: bool
    status: str
    source_path: str
    filename: str

    page_count: int = 0
    pages: List[PDFPage] = field(default_factory=list)

    text: str = ""
    metadata: dict = field(default_factory=dict)

    requires_ocr: bool = False
    ocr_pages: List[int] = field(default_factory=list)

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_NOT_FOUND = "not_found"
STATUS_PATH_DENIED = "path_denied"
STATUS_UNSUPPORTED_TYPE = "unsupported_type"
STATUS_EMPTY = "empty"
STATUS_REQUIRES_OCR = "requires_ocr"
STATUS_MALFORMED = "malformed"


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

def _is_within_allowed_roots(path: Path) -> bool:
    """
    Keep PDF access consistent with knowledge_base.ingestion.py.
    """

    return (
        path.is_relative_to(_DOCUMENTS_ROOT)
        or path.is_relative_to(_UPLOADS_ROOT)
    )


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text without changing its meaning.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    # Normalize horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _has_meaningful_text(text: str) -> bool:
    """
    Determine whether a page contains enough extracted text to be
    considered a real text layer rather than empty/noise.
    """

    if not text:
        return False

    compact = re.sub(r"\s+", "", text)

    return len(compact) >= MIN_TEXT_CHARS


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Local PDF parser.

    It performs text-layer extraction and modality detection only.
    OCR is intentionally delegated to ocr_pipeline.py.
    """

    def __init__(
        self,
        *,
        min_text_chars: int = MIN_TEXT_CHARS,
    ) -> None:

        if min_text_chars <= 0:
            raise ValueError("min_text_chars must be positive.")

        self.min_text_chars = min_text_chars

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def parse(
        self,
        path: Union[str, Path],
    ) -> PDFParseResult:
        """
        Parse one PDF and determine whether OCR is required.

        Expected failures are returned as structured results instead of
        being propagated to the API layer.
        """

        raw_path = Path(path)
        filename = raw_path.name

        def fail(
            status: str,
            error: str,
            source_path: Optional[str] = None,
        ) -> PDFParseResult:

            logger.warning(
                "PDF parsing failed for '%s': [%s] %s",
                raw_path,
                status,
                error,
            )

            return PDFParseResult(
                success=False,
                status=status,
                source_path=source_path or str(raw_path),
                filename=filename,
                error=error,
            )

        # ---------------------------------------------------------------
        # Resolve path
        # ---------------------------------------------------------------

        try:
            resolved = raw_path.resolve()
        except OSError as exc:
            return fail(
                STATUS_PATH_DENIED,
                f"Could not resolve path: {exc}",
            )

        # ---------------------------------------------------------------
        # Security boundary
        # ---------------------------------------------------------------

        if not _is_within_allowed_roots(resolved):
            return fail(
                STATUS_PATH_DENIED,
                (
                    "Path is outside the allowed "
                    "data/documents/ or data/uploads/ directories."
                ),
                str(resolved),
            )

        # ---------------------------------------------------------------
        # File validation
        # ---------------------------------------------------------------

        if resolved.suffix.lower() != ".pdf":
            return fail(
                STATUS_UNSUPPORTED_TYPE,
                f"Unsupported file type '{resolved.suffix or 'unknown'}'.",
                str(resolved),
            )

        if not resolved.is_file():
            return fail(
                STATUS_NOT_FOUND,
                "PDF file does not exist.",
                str(resolved),
            )

        # ---------------------------------------------------------------
        # Open PDF
        # ---------------------------------------------------------------

        try:
            reader = pypdf.PdfReader(str(resolved))

            if reader.is_encrypted:
                decrypted = reader.decrypt("")

                if not decrypted:
                    return fail(
                        STATUS_MALFORMED,
                        "PDF is password-protected and could not be opened.",
                        str(resolved),
                    )

        except Exception as exc:
            return fail(
                STATUS_MALFORMED,
                f"Failed to open PDF: {exc}",
                str(resolved),
            )

        # ---------------------------------------------------------------
        # Page extraction
        # ---------------------------------------------------------------

        pages: List[PDFPage] = []
        ocr_pages: List[int] = []
        page_texts: List[str] = []

        try:
            for index, page in enumerate(reader.pages, start=1):

                try:
                    extracted = page.extract_text() or ""
                except Exception as exc:
                    logger.warning(
                        "Text extraction failed on PDF page %d: %s",
                        index,
                        exc,
                    )
                    extracted = ""

                normalized = _normalize_text(extracted)

                has_text = len(
                    re.sub(r"\s+", "", normalized)
                ) >= self.min_text_chars

                requires_ocr = not has_text

                if requires_ocr:
                    ocr_pages.append(index)

                pages.append(
                    PDFPage(
                        page_number=index,
                        text=normalized,
                        has_text=has_text,
                        requires_ocr=requires_ocr,
                    )
                )

                page_texts.append(normalized)

        except Exception as exc:
            return fail(
                STATUS_MALFORMED,
                f"Failed while reading PDF pages: {exc}",
                str(resolved),
            )

        # ---------------------------------------------------------------
        # Empty PDF
        # ---------------------------------------------------------------

        page_count = len(pages)

        if page_count == 0:
            return fail(
                STATUS_EMPTY,
                "PDF contains no pages.",
                str(resolved),
            )

        # ---------------------------------------------------------------
        # Combined text
        # ---------------------------------------------------------------

        combined_text = "\n\n".join(
            text for text in page_texts if text
        ).strip()

        # ---------------------------------------------------------------
        # Modality detection
        # ---------------------------------------------------------------

        all_pages_require_ocr = (
            page_count > 0
            and len(ocr_pages) == page_count
        )

        some_pages_require_ocr = (
            len(ocr_pages) > 0
            and not all_pages_require_ocr
        )

        # ---------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------

        metadata = {}

        try:
            raw_metadata = reader.metadata

            if raw_metadata:
                for key, value in raw_metadata.items():
                    if value is not None:
                        metadata[str(key)] = str(value)

        except Exception as exc:
            logger.debug(
                "Could not read PDF metadata for '%s': %s",
                resolved,
                exc,
            )

        metadata.update(
            {
                "page_count": page_count,
                "text_pages": page_count - len(ocr_pages),
                "ocr_pages": len(ocr_pages),
                "requires_ocr": bool(ocr_pages),
            }
        )

        # ---------------------------------------------------------------
        # Scanned / mixed / text PDF
        # ---------------------------------------------------------------

        if all_pages_require_ocr:
            logger.info(
                "PDF '%s' is image-only/scanned; OCR required for all pages.",
                resolved,
            )

            return PDFParseResult(
                success=False,
                status=STATUS_REQUIRES_OCR,
                source_path=str(resolved),
                filename=filename,
                page_count=page_count,
                pages=pages,
                text="",
                metadata=metadata,
                requires_ocr=True,
                ocr_pages=ocr_pages,
                error=(
                    "No meaningful extractable text was found. "
                    "Route the PDF through the OCR pipeline."
                ),
            )

        # ---------------------------------------------------------------
        # Mixed PDF
        # ---------------------------------------------------------------

        if some_pages_require_ocr:
            logger.info(
                "PDF '%s' contains %d page(s) requiring OCR.",
                resolved,
                len(ocr_pages),
            )

            return PDFParseResult(
                success=True,
                status="mixed",
                source_path=str(resolved),
                filename=filename,
                page_count=page_count,
                pages=pages,
                text=combined_text,
                metadata=metadata,
                requires_ocr=True,
                ocr_pages=ocr_pages,
                error=None,
            )

        # ---------------------------------------------------------------
        # Normal text PDF
        # ---------------------------------------------------------------

        if not combined_text:
            return PDFParseResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(resolved),
                filename=filename,
                page_count=page_count,
                pages=pages,
                text="",
                metadata=metadata,
                requires_ocr=False,
                ocr_pages=[],
                error="No text content found in PDF.",
            )

        logger.info(
            "Parsed text PDF '%s' (%d page(s)).",
            resolved,
            page_count,
        )

        return PDFParseResult(
            success=True,
            status=STATUS_SUCCESS,
            source_path=str(resolved),
            filename=filename,
            page_count=page_count,
            pages=pages,
            text=combined_text,
            metadata=metadata,
            requires_ocr=False,
            ocr_pages=[],
            error=None,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse_pdf(
    path: Union[str, Path],
) -> PDFParseResult:
    """
    Convenience wrapper around PDFParser.parse().
    """

    return PDFParser().parse(path)