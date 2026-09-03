"""
Tests for backend.multimodel.vision_pipeline

These tests do NOT load the real Moondream model.
A fake model is injected so the pipeline logic can be tested
without downloading model weights or requiring GPU inference.
"""

from pathlib import Path

import pytest
from PIL import Image

from backend.multimodel.vision_pipeline import (
    VisionPipeline,
    VisionDetection,
    STATUS_SUCCESS,
    STATUS_NOT_FOUND,
    STATUS_PATH_DENIED,
    STATUS_UNSUPPORTED_TYPE,
    STATUS_EMPTY,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

@pytest.fixture
def allowed_dir(tmp_path, monkeypatch):
    """
    Create a temporary directory and make it the allowed
    documents directory for the pipeline.
    """
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    monkeypatch.setattr(
        "backend.multimodel.vision_pipeline._DOCUMENTS_ROOT",
        documents_dir.resolve(),
    )

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    monkeypatch.setattr(
        "backend.multimodel.vision_pipeline._UPLOADS_ROOT",
        upload_dir.resolve(),
    )

    return documents_dir


@pytest.fixture
def sample_image(allowed_dir):
    """Create a small valid test image."""
    image_path = allowed_dir / "test_image.png"

    image = Image.new("RGB", (200, 100), "white")
    image.save(image_path)

    return image_path


class FakeVisionModel:
    """
    Fake Moondream-compatible model.

    Used so tests don't download/load the real model.
    """

    def caption(self, image, length="normal"):
        return {
            "caption": "A white industrial inspection image."
        }

    def query(self, image, question):
        return {
            "answer": "The image appears to contain an industrial component."
        }

    def detect(self, image, object_query):
        return {
            "objects": [
                {
                    "x_min": 0.10,
                    "y_min": 0.20,
                    "x_max": 0.80,
                    "y_max": 0.90,
                }
            ]
        }


@pytest.fixture
def pipeline():
    """
    Create pipeline and inject fake model.
    """
    pipeline = VisionPipeline(
        model_id="fake-model",
        device="cpu",
    )

    pipeline._model = FakeVisionModel()

    return pipeline


# -------------------------------------------------------------------
# Validation Tests
# -------------------------------------------------------------------

def test_valid_image_validation(pipeline, sample_image):
    """Valid image inside allowed directory should pass validation."""

    path, error = pipeline._validate_image(sample_image)

    assert error is None
    assert path is not None
    assert path.exists()
   # assert path.suffix.lower() == ".png"


def test_file_not_found(pipeline, allowed_dir):
    """Non-existing image should return not_found."""

    missing_file = allowed_dir / "does_not_exist.png"

    result = pipeline.caption(missing_file)

    assert result.success is False
    assert result.status == STATUS_NOT_FOUND


def test_unsupported_file_type(pipeline, allowed_dir):
    """Unsupported extension should be rejected."""

    txt_file = allowed_dir / "test.txt"
    txt_file.write_text("not an image")

    result = pipeline.caption(txt_file)

    assert result.success is False
    assert result.status == STATUS_UNSUPPORTED_TYPE


def test_path_traversal_is_blocked(pipeline, tmp_path):
    """Image outside allowed directories must be rejected."""

    outside_file = tmp_path / "outside.png"

    image = Image.new("RGB", (100, 100), "white")
    image.save(outside_file)

    result = pipeline.caption(outside_file)

    assert result.success is False
    assert result.status == STATUS_PATH_DENIED


# -------------------------------------------------------------------
# Caption Tests
# -------------------------------------------------------------------

def test_caption_success(pipeline, sample_image):
    """Caption operation should return successful result."""

    result = pipeline.caption(sample_image)

    assert result.success is True
    assert result.status == STATUS_SUCCESS
    assert result.operation == "caption"

    assert result.caption is not None
    assert "industrial" in result.caption.lower()

    assert result.filename == "test_image.png"


def test_caption_length_is_passed_to_model(pipeline, sample_image):
    """Caption should pass requested length to the model."""

    calls = {}

    class FakeCaptionModel:
        def caption(self, image, length="normal"):
            calls["length"] = length

            return {
                "caption": "Test caption"
            }

    pipeline._model = FakeCaptionModel()

    result = pipeline.caption(
        sample_image,
        length="short",
    )

    assert result.success is True
    assert calls["length"] == "short"


# -------------------------------------------------------------------
# Query Tests
# -------------------------------------------------------------------

def test_query_success(pipeline, sample_image):
    """Question answering should return model answer."""

    question = "What is visible in the image?"

    result = pipeline.query(
        sample_image,
        question,
    )

    assert result.success is True
    assert result.status == STATUS_SUCCESS
    assert result.operation == "query"

    assert result.question == question
    assert result.answer is not None
    assert "industrial" in result.answer.lower()


def test_query_empty_question(pipeline, sample_image):
    """Empty question should be rejected."""

    result = pipeline.query(
        sample_image,
        "",
    )

    assert result.success is False
    assert result.status == STATUS_EMPTY


def test_query_whitespace_question(pipeline, sample_image):
    """Whitespace-only question should be rejected."""

    result = pipeline.query(
        sample_image,
        "   ",
    )

    assert result.success is False
    assert result.status == STATUS_EMPTY


# -------------------------------------------------------------------
# Detection Tests
# -------------------------------------------------------------------

def test_detect_success(pipeline, sample_image):
    """Object detection should return normalized bounding box."""

    result = pipeline.detect(
        sample_image,
        "machine",
    )

    assert result.success is True
    assert result.status == STATUS_SUCCESS
    assert result.operation == "detect"

    assert result.object_query == "machine"

    assert len(result.detections) == 1

    detection = result.detections[0]

    assert isinstance(detection, VisionDetection)

    assert detection.x_min == pytest.approx(0.10)
    assert detection.y_min == pytest.approx(0.20)
    assert detection.x_max == pytest.approx(0.80)
    assert detection.y_max == pytest.approx(0.90)


def test_detect_empty_object_query(pipeline, sample_image):
    """Empty object query should be rejected."""

    result = pipeline.detect(
        sample_image,
        "",
    )

    assert result.success is False
    assert result.status == STATUS_EMPTY


def test_detect_invalid_coordinates_are_ignored(
    pipeline,
    sample_image,
):
    """Invalid bounding boxes should not be accepted."""

    class BadDetectionModel:

        def detect(self, image, object_query):
            return {
                "objects": [
                    {
                        "x_min": -0.5,
                        "y_min": 0.2,
                        "x_max": 2.0,
                        "y_max": 0.9,
                    }
                ]
            }

    pipeline._model = BadDetectionModel()

    result = pipeline.detect(
        sample_image,
        "machine",
    )

    assert result.success is True
    assert len(result.detections) == 0


# -------------------------------------------------------------------
# Model Loading Tests
# -------------------------------------------------------------------

def test_existing_model_is_not_loaded_again(
    pipeline,
    sample_image,
    monkeypatch,
):
    """
    If a model is already loaded, _load_model should not
    unnecessarily load it again.
    """

    fake_model = FakeVisionModel()

    pipeline._model = fake_model

    called = False

    def fake_loader():
        nonlocal called
        called = True

    monkeypatch.setattr(
        pipeline,
        "_load_model",
        fake_loader,
    )

    result = pipeline.caption(sample_image)

    assert result.success is True
    assert called is False


# -------------------------------------------------------------------
# Metadata Tests
# -------------------------------------------------------------------

def test_caption_metadata(pipeline, sample_image):
    """Result should contain useful image metadata."""

    result = pipeline.caption(sample_image)

    assert result.success is True

    assert "width" in result.metadata
    assert "height" in result.metadata
    assert "format" in result.metadata

    assert result.metadata["width"] == 200
    assert result.metadata["height"] == 100


# -------------------------------------------------------------------
# Multiple Operations
# -------------------------------------------------------------------

def test_all_operations_on_same_image(
    pipeline,
    sample_image,
):
    """Caption, query and detection should all work."""

    caption_result = pipeline.caption(sample_image)

    query_result = pipeline.query(
        sample_image,
        "What is shown?",
    )

    detection_result = pipeline.detect(
        sample_image,
        "component",
    )

    assert caption_result.success is True
    assert query_result.success is True
    assert detection_result.success is True

    assert caption_result.operation == "caption"
    assert query_result.operation == "query"
    assert detection_result.operation == "detect"