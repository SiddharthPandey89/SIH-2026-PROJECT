"""
backend/multimodal/vision_pipeline.py

Local vision pipeline for the Sovereign AI Workbench.

Responsibilities:
    - Validate local image paths.
    - Load images safely.
    - Run local Moondream2 inference.
    - Support image captioning.
    - Support visual question answering.
    - Support object detection.
    - Return structured results.

This module does NOT:
    - Perform OCR.
    - Build embeddings.
    - Use BGE-M3.
    - Use ChromaDB.
    - Perform vector search.
    - Call cloud APIs.
    - Execute arbitrary code.

OCR of scanned PDFs belongs to ocr_pipeline.py.
PDF parsing belongs to pdf_parser.py.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "VisionDetection",
    "VisionResult",
    "VisionPipeline",
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


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


DEFAULT_MODEL_ID = "vikhyatk/moondream2"


# ---------------------------------------------------------------------------
# Status values
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_NOT_FOUND = "not_found"
STATUS_PATH_DENIED = "path_denied"
STATUS_UNSUPPORTED_TYPE = "unsupported_type"
STATUS_MALFORMED = "malformed"
STATUS_MODEL_UNAVAILABLE = "model_unavailable"
STATUS_EMPTY = "empty"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class VisionDetection:
    """
    One detected object.

    Coordinates are normalized to the image dimensions:
        0.0 = left/top edge
        1.0 = right/bottom edge
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class VisionResult:
    """
    Structured result from the vision pipeline.
    """

    success: bool
    status: str

    source_path: str
    filename: str

    operation: str

    caption: Optional[str] = None
    answer: Optional[str] = None

    question: Optional[str] = None
    object_query: Optional[str] = None

    detections: List[VisionDetection] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_within_allowed_roots(path: Path) -> bool:
    return (
        path.is_relative_to(_DOCUMENTS_ROOT)
        or path.is_relative_to(_UPLOADS_ROOT)
    )


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def _safe_coordinate(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value < 0.0 or value > 1.0:
        return None

    return value


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------

class VisionPipeline:
    """
    Local Moondream2 vision pipeline.

    The model is loaded lazily so importing the module does not immediately
    consume GPU memory.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
    ) -> None:

        if not model_id:
            raise ValueError(
                "model_id must not be empty."
            )

        self.model_id = model_id
        self.device = device

        self._model = None
        self._image_class = None

    # -----------------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load Moondream2 locally using Transformers.

        No API key or cloud endpoint is used.
        """

        if self._model is not None:
            return

        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM

        except ImportError as exc:
            raise RuntimeError(
                "Vision dependencies are missing. Install "
                "transformers, torch and Pillow."
            ) from exc

        self._image_class = Image

        if self.device == "cuda":
            device_map = "cuda"

        elif self.device == "cpu":
            device_map = "cpu"

        else:
            device_map = "auto"

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=device_map,
            dtype=dtype,
        )

    # -----------------------------------------------------------------------
    # Path validation
    # -----------------------------------------------------------------------

    def _validate_image(
        self,
        path: Union[str, Path],
    ) -> tuple[Optional[Path], Optional[VisionResult]]:

        raw_path = Path(path)

        filename = raw_path.name

        try:
            resolved = raw_path.resolve()

        except OSError as exc:

            return None, VisionResult(
                success=False,
                status=STATUS_PATH_DENIED,
                source_path=str(raw_path),
                filename=filename,
                operation="validate",
                error=f"Could not resolve path: {exc}",
            )

        if not _is_within_allowed_roots(resolved):

            return None, VisionResult(
                success=False,
                status=STATUS_PATH_DENIED,
                source_path=str(resolved),
                filename=filename,
                operation="validate",
                error=(
                    "Path is outside the allowed "
                    "data/documents/ or data/uploads/ directories."
                ),
            )

        if resolved.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:

            return None, VisionResult(
                success=False,
                status=STATUS_UNSUPPORTED_TYPE,
                source_path=str(resolved),
                filename=filename,
                operation="validate",
                error=(
                    f"Unsupported image type "
                    f"'{resolved.suffix or 'unknown'}'."
                ),
            )

        if not resolved.is_file():

            return None, VisionResult(
                success=False,
                status=STATUS_NOT_FOUND,
                source_path=str(resolved),
                filename=filename,
                operation="validate",
                error="Image file does not exist.",
            )

        return resolved, None

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------

    def _open_image(
        self,
        path: Path,
    ):
        try:
            from PIL import Image

            image = Image.open(path)

            # Force actual image decoding before inference.
            image.load()

            # Moondream works with RGB images.
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as exc:

            raise RuntimeError(
                f"Failed to open image: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Caption
    # -----------------------------------------------------------------------

    def caption(
        self,
        path: Union[str, Path],
        *,
        length: str = "normal",
    ) -> VisionResult:

        if length not in {
            "short",
            "normal",
            "long",
        }:
            return VisionResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(path),
                filename=Path(path).name,
                operation="caption",
                error=(
                    "length must be 'short', "
                    "'normal' or 'long'."
                ),
            )

        image_path, validation_error = (
            self._validate_image(path)
        )

        if validation_error:
            validation_error.operation = "caption"
            return validation_error

        assert image_path is not None

        try:
            image = self._open_image(image_path)
            
            if self._model is None:
                self._load_model()

            result = self._model.caption(
                image,
                length=length,
            )

            caption = ""

            if isinstance(result, dict):
                caption = result.get(
                    "caption",
                    "",
                )
            else:
                caption = getattr(
                    result,
                    "caption",
                    "",
                )

            caption = _normalize_text(caption)

            if not caption:
                return VisionResult(
                    success=False,
                    status=STATUS_EMPTY,
                    source_path=str(image_path),
                    filename=image_path.name,
                    operation="caption",
                    error="Model returned an empty caption.",
                )

            return VisionResult(
                success=True,
                status=STATUS_SUCCESS,
                source_path=str(image_path),
                filename=image_path.name,
                operation="caption",
                caption=caption,
                metadata={
                    "model": self.model_id,
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                },
            )

        except Exception as exc:

            logger.exception(
                "Vision captioning failed for '%s'.",
                image_path,
            )

            return VisionResult(
                success=False,
                status=STATUS_MODEL_UNAVAILABLE,
                source_path=str(image_path),
                filename=image_path.name,
                operation="caption",
                error=str(exc),
            )

    # -----------------------------------------------------------------------
    # Visual question answering
    # -----------------------------------------------------------------------

    def query(
        self,
        path: Union[str, Path],
        question: str,
    ) -> VisionResult:
        if not isinstance(question, str):
            return VisionResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(path),
                filename=Path(path).name,
                operation="query",
                error="question must be string.",
            )

        question = _normalize_text(question)

        if not question:
            return VisionResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(path),
                filename=Path(path).name,
                operation="query",
                error="question must not be empty.",
            )

        image_path, validation_error = (
            self._validate_image(path)
        )

        if validation_error:
            validation_error.operation = "query"
            return validation_error

        assert image_path is not None

        try:
            image = self._open_image(image_path)

            if self._model is None:
                self._load_model()

            result = self._model.query(
                image,
                question,
            )

            answer = ""

            if isinstance(result, dict):
                answer = result.get(
                    "answer",
                    "",
                )
            else:
                answer = getattr(
                    result,
                    "answer",
                    "",
                )

            answer = _normalize_text(answer)

            if not answer:
                return VisionResult(
                    success=False,
                    status=STATUS_EMPTY,
                    source_path=str(image_path),
                    filename=image_path.name,
                    operation="query",
                    question=question,
                    error="Model returned an empty answer.",
                )

            return VisionResult(
                success=True,
                status=STATUS_SUCCESS,
                source_path=str(image_path),
                filename=image_path.name,
                operation="query",
                question=question,
                answer=answer,
                metadata={
                    "model": self.model_id,
                    "width": image.width,
                    "height": image.height,
                },
            )

        except Exception as exc:

            logger.exception(
                "Vision query failed for '%s'.",
                image_path,
            )

            return VisionResult(
                success=False,
                status=STATUS_MODEL_UNAVAILABLE,
                source_path=str(image_path),
                filename=image_path.name,
                operation="query",
                question=question,
                error=str(exc),
            )

    # -----------------------------------------------------------------------
    # Object detection
    # -----------------------------------------------------------------------

    def detect(
        self,
        path: Union[str, Path],
        object_query: str,
    ) -> VisionResult:
        if not isinstance(object_query, str):
            return VisionResult(
                success=False,
                status=STATUS_MALFORMED,
                source_path=str(path),
                filename=Path(path).name,
                operation="detect",
                error="object_query must be string.",
            )


        object_query = _normalize_text(object_query)
        

        if not object_query:
            return VisionResult(
                success=False,
                status=STATUS_EMPTY,
                source_path=str(path),
                filename=Path(path).name,
                operation="detect",
                error="object_query must not be empty.",
            )

        image_path, validation_error = (
            self._validate_image(path)
        )

        if validation_error:
            validation_error.operation = "detect"
            return validation_error

        assert image_path is not None

        try:
            image = self._open_image(image_path)

            if self._model is None:
                self._load_model()

            result = self._model.detect(
                image,
                object_query,
            )

            if isinstance(result, dict):
                objects = result.get(
                    "objects",
                    [],
                )
            else:
                objects = getattr(
                    result,
                    "objects",
                    [],
                )

            detections: List[VisionDetection] = []

            for obj in objects:

                if isinstance(obj, dict):
                    x_min = obj.get("x_min")
                    y_min = obj.get("y_min")
                    x_max = obj.get("x_max")
                    y_max = obj.get("y_max")

                else:
                    x_min = getattr(
                        obj,
                        "x_min",
                        None,
                    )
                    y_min = getattr(
                        obj,
                        "y_min",
                        None,
                    )
                    x_max = getattr(
                        obj,
                        "x_max",
                        None,
                    )
                    y_max = getattr(
                        obj,
                        "y_max",
                        None,
                    )

                x_min = _safe_coordinate(x_min)
                y_min = _safe_coordinate(y_min)
                x_max = _safe_coordinate(x_max)
                y_max = _safe_coordinate(y_max)

                if None in {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                }:
                    continue

                if x_min > x_max or y_min > y_max:
                    continue

                detections.append(
                    VisionDetection(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                )

            return VisionResult(
                success=True,
                status=STATUS_SUCCESS,
                source_path=str(image_path),
                filename=image_path.name,
                operation="detect",
                object_query=object_query,
                detections=detections,
                metadata={
                    "model": self.model_id,
                    "width": image.width,
                    "height": image.height,
                    "detection_count": len(detections),
                },
            )

        except Exception as exc:

            logger.exception(
                "Vision detection failed for '%s'.",
                image_path,
            )

            return VisionResult(
                success=False,
                status=STATUS_MODEL_UNAVAILABLE,
                source_path=str(image_path),
                filename=image_path.name,
                operation="detect",
                object_query=object_query,
                error=str(exc),
            )