"""
backend/multimodal/ocr_pipeline.py

Local OCR pipeline for the Sovereign AI Workbench.

Responsibilities:
    - Accept a local PDF.
    - Render selected PDF pages to images.
    - Run Surya OCR locally.
    - Preserve page numbers.
    - Preserve OCR confidence where available.
    - Return structured OCR results.

This module does NOT:
    - Build embeddings.
    - Use BGE-M3.
    - Build/search ChromaDB.
    - Perform vector retrieval.
    - Call cloud APIs.
    - Generate final documents.
    - Execute arbitrary code.

PDF modality detection remains the responsibility of pdf_parser.py.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pypdf

logger = logging.getLogger(__name__)

__all__ = [
    "OCRBlock",
    "OCRPageResult",
    "OCRResult",
    "OCRPipeline",
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

DEFAULT_DPI = 150


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class OCRBlock:
    """
    One recognized OCR block.
    """

    text: str
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None


@dataclass
class OCRPageResult:
    """
    OCR result for one PDF page.
    """

    page_number: int
    text: str
    blocks: List[OCRBlock] = field(default_factory=list)

    confidence: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class OCRResult:
    """
    Complete OCR pipeline result.
    """

    success: bool
    status: str

    source_path: str
    filename: str

    page_count: int = 0
    pages: List[OCRPageResult] = field(default_factory=list)

    text: str = ""

    processed_pages: List[int] = field(default_factory=list)
    failed_pages: List[int] = field(default_factory=list)

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_NOT_FOUND = "not_found"
STATUS_PATH_DENIED = "path_denied"
STATUS_UNSUPPORTED_TYPE = "unsupported_type"
STATUS_MALFORMED = "malformed"
STATUS_EMPTY = "empty"
STATUS_OCR_UNAVAILABLE = "ocr_unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_within_allowed_roots(path: Path) -> bool:
    return (
        path.is_relative_to(_DOCUMENTS_ROOT)
        or path.is_relative_to(_UPLOADS_ROOT)
    )


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if 0.0 <= result <= 1.0:
            return result

    except (TypeError, ValueError):
        pass

    return None


def _safe_bbox(value) -> Optional[List[float]]:
    if value is None:
        return None

    try:
        values = list(value)

        if len(values) != 4:
            return None

        return [float(v) for v in values]

    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# OCR pipeline
# ---------------------------------------------------------------------------

class OCRPipeline:
    """
    Local Surya OCR pipeline.

    Surya is loaded lazily so importing this module does not immediately
    load a large model into memory.
    """

    def __init__(
        self,
        *,
        dpi: int = DEFAULT_DPI,
    ) -> None:

        if dpi <= 0:
            raise ValueError("dpi must be a positive integer.")

        self.dpi = dpi
        self._manager = None
        self._recognition_predictor = None

    # -----------------------------------------------------------------------
    # Lazy Surya initialization
    # -----------------------------------------------------------------------

    def _load_surya(self) -> None:
        """
        Initialize the current Surya inference backend lazily.

        Current Surya versions use SuryaInferenceManager and
        RecognitionPredictor.
        """

        if self._recognition_predictor is not None:
            return

        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

        except ImportError as exc:
            raise RuntimeError(
                "Surya OCR is not installed. "
                "Install the required local OCR dependency with "
                "'pip install surya-ocr'."
            ) from exc

        self._manager = SuryaInferenceManager()

        self._recognition_predictor = RecognitionPredictor(
            self._manager
        )

    # -----------------------------------------------------------------------
    # PDF validation
    # -----------------------------------------------------------------------

    def _validate_pdf(
        self,
        path: Union[str, Path],
    ) -> tuple[Optional[Path], Optional[OCRResult]]:

        raw_path = Path(path)
        filename = raw_path.name

        try:
            resolved = raw_path.resolve()
        except OSError as exc:
            return None, OCRResult(
                success=False,
                status=STATUS_PATH_DENIED,
                source_path=str(raw_path),
                filename=filename,
                error=f"Could not resolve path: {exc}",
            )

        if not _is_within_allowed_roots(resolved):
            return None, OCRResult(
                success=False,
                status=STATUS_PATH_DENIED,
                source_path=str(resolved),
                filename=filename,
                error=(
                    "Path is outside the allowed "
                    "data/documents/ or data/uploads/ directories."
                ),
            )

        if resolved.suffix.lower() != ".pdf":
            return None, OCRResult(
                success=False,
                status=STATUS_UNSUPPORTED_TYPE,
                source_path=str(resolved),
                filename=filename,
                error="OCR pipeline accepts PDF files only.",
            )

        if not resolved.is_file():
            return None, OCRResult(
                success=False,
                status=STATUS_NOT_FOUND,
                source_path=str(resolved),
                filename=filename,
                error="PDF file does not exist.",
            )

        return resolved, None

    # -----------------------------------------------------------------------
    # PDF → images
    # -----------------------------------------------------------------------

    def _render_pages(
        self,
        pdf_path: Path,
        page_numbers: Sequence[int],
    ):
        """
        Render selected PDF pages into PIL images.

        Uses PyMuPDF when available.
        """

        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to render PDF pages for OCR. "
                "Install it with 'pip install pymupdf'."
            ) from exc

        document = fitz.open(str(pdf_path))

        try:
            total_pages = len(document)

            for page_number in page_numbers:

                if page_number < 1 or page_number > total_pages:
                    raise ValueError(
                        f"Invalid page number: {page_number}"
                    )

                page = document.load_page(page_number - 1)

                scale = self.dpi / 72.0

                matrix = fitz.Matrix(
                    scale,
                    scale,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image = pixmap.pil_image()

                yield page_number, image

        finally:
            document.close()

    # -----------------------------------------------------------------------
    # Extract Surya result
    # -----------------------------------------------------------------------

    def _extract_prediction(
        self,
        prediction,
    ) -> tuple[str, List[OCRBlock], Optional[float]]:
        """
        Convert Surya's prediction object into our stable internal format.

        This keeps the rest of the application independent from Surya's
        exact result classes.
        """

        blocks: List[OCRBlock] = []

        raw_blocks = getattr(
            prediction,
            "blocks",
            None,
        )

        if raw_blocks is None and isinstance(prediction, dict):
            raw_blocks = prediction.get("blocks")

        if raw_blocks is None:
            raw_blocks = []

        texts: List[str] = []
        confidences: List[float] = []

        for raw_block in raw_blocks:

            text = getattr(raw_block, "text", None)

            if text is None and isinstance(raw_block, dict):
                text = raw_block.get("text")

            if text is None:
                text = getattr(raw_block, "html", "")

                if text is None and isinstance(raw_block, dict):
                    text = raw_block.get("html", "")

            text = _normalize_text(str(text or ""))

            if not text:
                continue

            confidence = getattr(
                raw_block,
                "confidence",
                None,
            )

            if confidence is None and isinstance(raw_block, dict):
                confidence = raw_block.get("confidence")

            confidence = _safe_float(confidence)

            bbox = getattr(
                raw_block,
                "bbox",
                None,
            )

            if bbox is None and isinstance(raw_block, dict):
                bbox = raw_block.get("bbox")

            bbox = _safe_bbox(bbox)

            blocks.append(
                OCRBlock(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

            texts.append(text)

            if confidence is not None:
                confidences.append(confidence)

        combined_text = _normalize_text(
            "\n".join(texts)
        )

        page_confidence = None

        if confidences:
            page_confidence = sum(confidences) / len(confidences)

        return (
            combined_text,
            blocks,
            page_confidence,
        )

    # -----------------------------------------------------------------------
    # Public OCR API
    # -----------------------------------------------------------------------

    def process(
        self,
        path: Union[str, Path],
        *,
        page_numbers: Optional[Sequence[int]] = None,
    ) -> OCRResult:
        """
        Run OCR on the requested PDF pages.

        If page_numbers is None, every page is processed.
        """

        pdf_path, validation_error = self._validate_pdf(path)

        if validation_error is not None:
            return validation_error

        assert pdf_path is not None

        filename = pdf_path.name

        # ---------------------------------------------------------------
        # Read PDF
        # ---------------------------------------------------------------

        try:
            reader = pypdf.PdfReader(str(pdf_path))

            if reader.is_encrypted:
                decrypted = reader.decrypt("")

                if not decrypted:
                    return OCRResult(
                        success=False,
                        status=STATUS_MALFORMED,
                        source_path=str(pdf_path),
                        filename=filename,
                        error=(
                            "PDF is password-protected and "
                            "could not be opened."
                        ),
                    )

            total_pages = len(reader.pages)

        except Exception as exc:
            return OCRResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(pdf_path),
                filename=filename,
                error=f"Failed to open PDF: {exc}",
            )

        if total_pages == 0:
            return OCRResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(pdf_path),
                filename=filename,
                error="PDF contains no pages.",
            )

        # ---------------------------------------------------------------
        # Select pages
        # ---------------------------------------------------------------

        if page_numbers is None:
            selected_pages = list(
                range(1, total_pages + 1)
            )
        else:
            selected_pages = sorted(
                set(int(page) for page in page_numbers)
            )

        invalid_pages = [
            page
            for page in selected_pages
            if page < 1 or page > total_pages
        ]

        if invalid_pages:
            return OCRResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                error=(
                    f"Invalid page number(s): {invalid_pages}"
                ),
            )

        if not selected_pages:
            return OCRResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                error="No pages selected for OCR.",
            )

        # ---------------------------------------------------------------
        # Load Surya
        # ---------------------------------------------------------------

        try:
            self._load_surya()

        except Exception as exc:
            logger.exception("Could not initialize Surya OCR.")

            return OCRResult(
                success=False,
                status=STATUS_OCR_UNAVAILABLE,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                error=str(exc),
            )

        # ---------------------------------------------------------------
        # Render pages
        # ---------------------------------------------------------------

        rendered_pages = []

        try:
            for page_number, image in self._render_pages(
                pdf_path,
                selected_pages,
            ):
                rendered_pages.append(
                    (page_number, image)
                )

        except Exception as exc:
            return OCRResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                error=f"Failed to render PDF pages: {exc}",
            )

        # ---------------------------------------------------------------
        # Run OCR
        # ---------------------------------------------------------------

        results: List[OCRPageResult] = []
        processed_pages: List[int] = []
        failed_pages: List[int] = []

        try:
            images = [
                image
                for _, image in rendered_pages
            ]

            predictions = self._recognition_predictor(
                images
            )

        except Exception as exc:
            logger.exception("Surya OCR inference failed.")

            return OCRResult(
                success=False,
                status=STATUS_OCR_UNAVAILABLE,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                error=f"Surya OCR inference failed: {exc}",
            )

        # ---------------------------------------------------------------
        # Convert results
        # ---------------------------------------------------------------

        for (page_number, _), prediction in zip(
            rendered_pages,
            predictions,
        ):

            try:
                text, blocks, confidence = (
                    self._extract_prediction(prediction)
                )

                if text:
                    results.append(
                        OCRPageResult(
                            page_number=page_number,
                            text=text,
                            blocks=blocks,
                            confidence=confidence,
                            success=True,
                        )
                    )

                    processed_pages.append(page_number)

                else:
                    results.append(
                        OCRPageResult(
                            page_number=page_number,
                            text="",
                            blocks=[],
                            confidence=None,
                            success=False,
                            error="No text recognized on page.",
                        )
                    )

                    failed_pages.append(page_number)

            except Exception as exc:
                logger.exception(
                    "Failed to process OCR result for page %d.",
                    page_number,
                )

                results.append(
                    OCRPageResult(
                        page_number=page_number,
                        text="",
                        blocks=[],
                        confidence=None,
                        success=False,
                        error=str(exc),
                    )
                )

                failed_pages.append(page_number)

        # ---------------------------------------------------------------
        # Final combined text
        # ---------------------------------------------------------------

        combined_text = _normalize_text(
            "\n\n".join(
                page.text
                for page in results
                if page.text
            )
        )

        if not processed_pages:
            return OCRResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(pdf_path),
                filename=filename,
                page_count=total_pages,
                pages=results,
                text="",
                processed_pages=[],
                failed_pages=failed_pages,
                error="OCR completed but no text was recognized.",
            )

        if failed_pages:
            status = STATUS_PARTIAL
        else:
            status = STATUS_SUCCESS

        logger.info(
            "OCR completed for '%s': %d/%d page(s).",
            pdf_path,
            len(processed_pages),
            len(selected_pages),
        )

        return OCRResult(
            success=True,
            status=status,
            source_path=str(pdf_path),
            filename=filename,
            page_count=total_pages,
            pages=results,
            text=combined_text,
            processed_pages=processed_pages,
            failed_pages=failed_pages,
            error=None,
        )