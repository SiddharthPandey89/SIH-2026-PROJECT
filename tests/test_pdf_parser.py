"""
tests/test_pdf_parser.py

Tests for backend/multimodel/pdf_parser.py

Covers:
    - Normal text PDF
    - Multi-page PDF
    - Scanned/image-only PDF
    - Mixed PDF
    - Missing PDF
    - Unsupported file type
    - Path security
    - Empty PDF
    - PDF metadata
"""

from pathlib import Path

from backend.multimodel.pdf_parser import PDFParser


# ---------------------------------------------------------------------------
# Test data directory
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path("data/documents")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_test_file(name: str) -> Path:
    """
    Return a test PDF from data/documents/.
    """
    return TEST_DATA_DIR / name


# ---------------------------------------------------------------------------
# Normal text PDF
# ---------------------------------------------------------------------------

def test_parse_normal_text_pdf():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_text.pdf")
    )

    assert result.success is True
    assert result.status == "success"

    assert result.filename == "test_text.pdf"
    assert result.page_count > 0

    assert result.text
    assert len(result.pages) == result.page_count

    assert result.requires_ocr is False
    assert result.ocr_pages == []

    for page in result.pages:
        assert page.page_number > 0
        assert page.has_text is True
        assert page.requires_ocr is False


# ---------------------------------------------------------------------------
# Multi-page PDF
# ---------------------------------------------------------------------------

def test_parse_multipage_pdf():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_multipage.pdf")
    )

    assert result.success is True
    assert result.status == "success"

    assert result.page_count >= 2
    assert len(result.pages) == result.page_count

    page_numbers = [
        page.page_number
        for page in result.pages
    ]

    assert page_numbers == list(
        range(1, result.page_count + 1)
    )


# ---------------------------------------------------------------------------
# Scanned PDF
# ---------------------------------------------------------------------------

def test_parse_scanned_pdf_requires_ocr():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_scanned.pdf")
    )

    assert result.success is False
    assert result.status == "requires_ocr"

    assert result.requires_ocr is True
    assert len(result.ocr_pages) > 0

    assert result.page_count > 0
    assert len(result.pages) == result.page_count

    for page in result.pages:
        assert page.has_text is False
        assert page.requires_ocr is True


# ---------------------------------------------------------------------------
# Mixed PDF
# ---------------------------------------------------------------------------

def test_parse_mixed_pdf():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_mixed.pdf")
    )

    assert result.success is True
    assert result.status == "mixed"

    assert result.requires_ocr is True

    assert len(result.ocr_pages) > 0
    assert len(result.ocr_pages) < result.page_count

    assert result.text

    for page in result.pages:
        if page.page_number in result.ocr_pages:
            assert page.requires_ocr is True
        else:
            assert page.has_text is True
            assert page.requires_ocr is False


# ---------------------------------------------------------------------------
# Missing PDF
# ---------------------------------------------------------------------------

def test_missing_pdf():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("does_not_exist.pdf")
    )

    assert result.success is False
    assert result.status == "not_found"

    assert result.text == ""
    assert result.page_count == 0


# ---------------------------------------------------------------------------
# Unsupported file type
# ---------------------------------------------------------------------------

def test_unsupported_file_type():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test.txt")
    )

    assert result.success is False
    assert result.status == "unsupported_type"

    assert result.text == ""
    assert result.page_count == 0


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

def test_path_outside_allowed_root():
    parser = PDFParser()

    result = parser.parse(
        Path("C:/Users/Public/test.pdf")
    )

    assert result.success is False
    assert result.status == "path_denied"

    assert result.text == ""
    assert result.page_count == 0


# ---------------------------------------------------------------------------
# Empty PDF
# ---------------------------------------------------------------------------

def test_empty_pdf():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_empty.pdf")
    )

    assert result.success is False
    assert result.status == "empty"

    assert result.page_count == 0
    assert result.text == ""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_pdf_metadata():
    parser = PDFParser()

    result = parser.parse(
        get_test_file("test_metadata.pdf")
    )

    assert result.success is True

    assert isinstance(result.metadata, dict)

    # Parser always adds these structural metadata fields.
    assert "page_count" in result.metadata
    assert "text_pages" in result.metadata
    assert "ocr_pages" in result.metadata
    assert "requires_ocr" in result.metadata

    assert result.metadata["page_count"] == result.page_count