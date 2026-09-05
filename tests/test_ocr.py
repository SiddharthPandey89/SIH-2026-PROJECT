"""
tests/test_ocr_pipeline.py

Tests for:
    backend/multimodel/ocr_pipeline.py

Testing strategy:
    - Fast unit tests do not load the Surya model.
    - Real OCR test is optional and runs only when explicitly enabled.

No:
    - BGE-M3
    - Vector store
    - ChromaDB
    - Network/cloud API
"""

from pathlib import Path

import pytest

from backend.multimodel.ocr_pipeline import (
    OCRPipeline,
    _normalize_text,
    _safe_bbox,
    _safe_float,
    OCRResult,
    OCRPageResult,
    OCRBlock,
)


TEST_DATA_DIR = Path("data/documents")


def get_test_file(name: str) -> Path:
    return TEST_DATA_DIR / name


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------

def test_ocr_block_model():
    block = OCRBlock(
        text="Hello World",
        confidence=0.95,
        bbox=[0.0, 0.0, 100.0, 50.0],
    )

    assert block.text == "Hello World"
    assert block.confidence == 0.95
    assert block.bbox == [0.0, 0.0, 100.0, 50.0]


def test_ocr_page_result_model():
    page = OCRPageResult(
        page_number=1,
        text="Safety Inspection Report",
        confidence=0.92,
    )

    assert page.page_number == 1
    assert page.text == "Safety Inspection Report"
    assert page.confidence == 0.92
    assert page.success is True
    assert page.error is None


def test_ocr_result_model():
    result = OCRResult(
        success=True,
        status="success",
        source_path="data/documents/test_scanned.pdf",
        filename="test_scanned.pdf",
        page_count=3,
    )

    assert result.success is True
    assert result.status == "success"
    assert result.page_count == 3
    assert result.pages == []
    assert result.text == ""


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

def test_pipeline_creation():
    pipeline = OCRPipeline()

    assert pipeline.dpi == 150
    assert pipeline._manager is None
    assert pipeline._recognition_predictor is None


def test_invalid_dpi():
    with pytest.raises(ValueError):
        OCRPipeline(dpi=0)


def test_negative_dpi():
    with pytest.raises(ValueError):
        OCRPipeline(dpi=-100)


# ---------------------------------------------------------------------------
# Missing PDF
# ---------------------------------------------------------------------------

def test_missing_pdf():
    pipeline = OCRPipeline()

    result = pipeline.process(
        get_test_file("does_not_exist.pdf")
    )

    assert result.success is False
    assert result.status == "not_found"
    assert result.page_count == 0


# ---------------------------------------------------------------------------
# Unsupported file
# ---------------------------------------------------------------------------

def test_unsupported_file():
    pipeline = OCRPipeline()

    result = pipeline.process(
        get_test_file("test.txt")
    )

    assert result.success is False
    assert result.status == "unsupported_type"


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

def test_path_outside_allowed_root():
    pipeline = OCRPipeline()

    result = pipeline.process(
        "C:/Users/Public/test.pdf"
    )

    assert result.success is False
    assert result.status == "path_denied"


# ---------------------------------------------------------------------------
# Empty PDF
# ---------------------------------------------------------------------------

def test_empty_pdf():
    pipeline = OCRPipeline()

    result = pipeline.process(
        get_test_file("test_empty.pdf")
    )

    assert result.success is False
    assert result.status == "empty"
    assert result.page_count == 0


# ---------------------------------------------------------------------------
# Invalid page number
# ---------------------------------------------------------------------------

def test_invalid_page_number(monkeypatch):
    """
    Prevent Surya from loading. We only test page validation here.
    """

    pipeline = OCRPipeline()

    result = pipeline.process(
        get_test_file("test_scanned.pdf"),
        page_numbers=[999999],
    )

    assert result.success is False
    assert result.status == "malformed"
    assert "Invalid page number" in result.error


# ---------------------------------------------------------------------------
# Empty page selection
# ---------------------------------------------------------------------------

def test_empty_page_selection():
    pipeline = OCRPipeline()

    result = pipeline.process(
        get_test_file("test_scanned.pdf"),
        page_numbers=[],
    )

    assert result.success is False
    assert result.status == "empty"
    assert result.error == "No pages selected for OCR."


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def test_normalize_text():
    pipeline = OCRPipeline()

    text = (
        "Hello    World\r\n"
        "\r\n"
        "\r\n"
        "Safety    Report"
    )

    normalized = _normalize_text(text)

    assert normalized == (
        "Hello World\n\nSafety Report"
    )


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def test_extract_prediction():
    pipeline = OCRPipeline()

    class FakeBlock:
        def __init__(self, text, confidence, bbox):
            self.text = text
            self.confidence = confidence
            self.bbox = bbox

    class FakePrediction:
        def __init__(self):
            self.blocks = [
                FakeBlock(
                    "Safety Report",
                    0.95,
                    [0, 0, 100, 50],
                ),
                FakeBlock(
                    "Inspection completed",
                    0.90,
                    [0, 60, 200, 100],
                ),
            ]

    text, blocks, confidence = (
        pipeline._extract_prediction(
            FakePrediction()
        )
    )

    assert text == (
        "Safety Report\nInspection completed"
    )

    assert len(blocks) == 2

    assert blocks[0].text == "Safety Report"
    assert blocks[0].confidence == 0.95

    assert blocks[1].text == "Inspection completed"
    assert blocks[1].confidence == 0.90

    assert confidence == pytest.approx(0.925)


# ---------------------------------------------------------------------------
# Empty OCR prediction
# ---------------------------------------------------------------------------

def test_empty_prediction():
    pipeline = OCRPipeline()

    class FakePrediction:
        blocks = []

    text, blocks, confidence = (
        pipeline._extract_prediction(
            FakePrediction()
        )
    )

    assert text == ""
    assert blocks == []
    assert confidence is None


# ---------------------------------------------------------------------------
# BBox validation
# ---------------------------------------------------------------------------

def test_invalid_bbox():
    pipeline = OCRPipeline()

    assert _safe_bbox(
        [1, 2, 3]
    ) is None

    assert _safe_bbox(
        None
    ) is None


def test_valid_bbox():
    pipeline = OCRPipeline()

    bbox = _safe_bbox(
        [1, 2, 100, 200]
    )

    assert bbox == [
        1.0,
        2.0,
        100.0,
        200.0,
    ]


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------

def test_valid_confidence():
    pipeline = OCRPipeline()

    assert _safe_float(0.95) == 0.95


def test_invalid_confidence():
    pipeline = OCRPipeline()

    assert _safe_float(None) is None
    assert _safe_float("abc") is None
    assert _safe_float(2.0) is None


# ---------------------------------------------------------------------------
# Real OCR test
# ---------------------------------------------------------------------------
#
# This test is intentionally skipped unless:
#
#     RUN_REAL_OCR_TEST=1
#
# is set.
#
# PowerShell:
#
#     $env:RUN_REAL_OCR_TEST="1"
#     python -m pytest tests/test_ocr_pipeline.py -v
#
# ---------------------------------------------------------------------------

@pytest.mark.real_ocr
def test_real_surya_ocr():
    import os

    if os.getenv("RUN_REAL_OCR_TEST") != "1":
        pytest.skip(
            "Real OCR test disabled. "
            "Set RUN_REAL_OCR_TEST=1 to run."
        )

    pdf_path = get_test_file(
        "test_scanned.pdf"
    )

    if not pdf_path.exists():
        pytest.fail(
            f"Test PDF not found: {pdf_path}"
        )

    pipeline = OCRPipeline()

    result = pipeline.process(pdf_path)

    assert result.page_count > 0

    assert result.success is True

    assert result.status in {
        "success",
        "partial",
    }

    assert len(result.pages) > 0

    assert len(result.processed_pages) > 0

    assert result.text

    for page in result.pages:
        assert page.page_number > 0
        assert isinstance(page.text, str)
        assert isinstance(
            page.blocks,
            list,
        )