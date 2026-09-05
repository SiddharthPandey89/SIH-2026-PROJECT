"""
tests/test_knowledge_base.py

Knowledge Base tests for the Sovereign AI Workbench.

Covers:
- Local document discovery
- Document chunking
- Keyword retrieval fallback
- Query relevance
- top_k behaviour
- Empty query handling
- Invalid top_k handling
- file_id filtering
- Result schema normalization
- Health check
- Vector-store backend contract
- Backend failure propagation

These tests are fully local/offline.
They do NOT require BGE-M3 or ChromaDB to be loaded.
"""

import pytest
from pathlib import Path

from backend.knowledge_base.retriever import (
    Retriever,
    VectorStoreBackend,
    _LocalFilesystemSearch,
    _tokenize,
    _score_chunk,
    _Chunk,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def knowledge_dirs(tmp_path, monkeypatch):
    """
    Create isolated temporary knowledge-base directories.
    """

    documents_dir = tmp_path / "documents"
    uploads_dir = tmp_path / "uploads"

    manuals_dir = documents_dir / "manuals"
    sops_dir = documents_dir / "sops"
    reports_dir = documents_dir / "past_reports"

    manuals_dir.mkdir(parents=True)
    sops_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "backend.knowledge_base.retriever._DOCUMENTS_DIR",
        documents_dir,
    )

    monkeypatch.setattr(
        "backend.knowledge_base.retriever._UPLOADS_DIR",
        uploads_dir,
    )

    return {
        "documents": documents_dir,
        "uploads": uploads_dir,
        "manuals": manuals_dir,
        "sops": sops_dir,
        "reports": reports_dir,
    }


@pytest.fixture
def sample_documents(knowledge_dirs):
    """
    Create realistic industrial/HSE test documents.
    """

    manual = knowledge_dirs["manuals"] / "pump_manual.txt"

    manual.write_text(
        """
        Centrifugal Pump Safety Manual

        Before starting the centrifugal pump, verify that the suction
        valve and discharge valve are in the correct operating position.

        Operators must wear appropriate personal protective equipment.
        Check for abnormal vibration, leakage, and unusual noise.

        Emergency shutdown must be performed immediately if unsafe
        operating conditions are detected.
        """,
        encoding="utf-8",
    )

    sop = knowledge_dirs["sops"] / "fire_safety_sop.md"

    sop.write_text(
        """
        # Fire Safety SOP

        Fire extinguishers must be inspected regularly.

        In case of an industrial fire, activate the emergency alarm,
        isolate the affected equipment if safe, and evacuate personnel
        according to the emergency response procedure.

        Only trained personnel should attempt fire suppression.
        """,
        encoding="utf-8",
    )

    report = knowledge_dirs["reports"] / "inspection_report.txt"

    report.write_text(
        """
        Inspection Report

        The inspection identified abnormal vibration near the pump
        assembly.

        Minor oil leakage was observed around the bearing housing.

        Corrective maintenance was recommended before continued operation.
        """,
        encoding="utf-8",
    )

    return {
        "manual": manual,
        "sop": sop,
        "report": report,
    }


@pytest.fixture
def retriever(knowledge_dirs, sample_documents):
    """
    Use deterministic local filesystem search.

    This avoids loading BGE-M3 / ChromaDB during unit tests.
    """

    backend = _LocalFilesystemSearch()

    return Retriever(backend=backend)


# -------------------------------------------------------------------
# Tokenization Tests
# -------------------------------------------------------------------

def test_tokenize_lowercases_text():
    """Tokenizer should normalize words to lowercase."""

    tokens = _tokenize(
        "Pump Safety Emergency"
    )

    assert tokens == [
        "pump",
        "safety",
        "emergency",
    ]


def test_tokenize_removes_punctuation():
    """Tokenizer should ignore punctuation."""

    tokens = _tokenize(
        "Pump, Safety! Emergency."
    )

    assert tokens == [
        "pump",
        "safety",
        "emergency",
    ]


def test_tokenize_empty_text():
    """Empty text should produce no tokens."""

    assert _tokenize("") == []


# -------------------------------------------------------------------
# Chunk Tests
# -------------------------------------------------------------------

def test_chunk_file(sample_documents):
    """Document should be converted into searchable chunks."""

    from backend.knowledge_base.retriever import _chunk_file

    chunks = _chunk_file(
        sample_documents["manual"]
    )

    assert len(chunks) > 0

    for chunk in chunks:
        assert isinstance(chunk, _Chunk)
        assert chunk.document_id
        assert chunk.title
        assert chunk.text


def test_chunk_title_generation(sample_documents):
    """Filename should be converted into a readable title."""

    from backend.knowledge_base.retriever import _chunk_file

    chunks = _chunk_file(
        sample_documents["manual"]
    )

    assert chunks[0].title == "Pump Manual"


# -------------------------------------------------------------------
# Scoring Tests
# -------------------------------------------------------------------

def test_chunk_score_relevant_text():
    """Relevant query terms should produce a positive score."""

    chunk = _Chunk(
        document_id="doc1",
        title="Pump Manual",
        text="Pump safety and emergency shutdown procedure",
    )

    query_terms = _tokenize(
        "pump safety"
    )

    score = _score_chunk(
        query_terms,
        chunk,
    )

    assert score > 0.0


def test_chunk_score_unrelated_text():
    """Unrelated content should have zero score."""

    chunk = _Chunk(
        document_id="doc1",
        title="Weather",
        text="Weather forecast and rainfall information",
    )

    query_terms = _tokenize(
        "pump safety"
    )

    score = _score_chunk(
        query_terms,
        chunk,
    )

    assert score == 0.0


# -------------------------------------------------------------------
# Retrieval Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieval_returns_relevant_document(
    retriever,
):
    """
    Query should retrieve the document containing the
    requested information.
    """

    results = await retriever.retrieve_context(
        query="pump safety",
        top_k=5,
    )

    assert len(results) > 0

    top_result = results[0]

    assert "document_id" in top_result
    assert "title" in top_result
    assert "snippet" in top_result
    assert "score" in top_result

    assert top_result["score"] > 0


@pytest.mark.asyncio
async def test_retrieval_fire_safety(
    retriever,
):
    """Fire-related query should retrieve the fire SOP."""

    results = await retriever.retrieve_context(
        query="fire emergency alarm evacuation",
        top_k=5,
    )

    assert len(results) > 0

    titles = [
        result["title"]
        for result in results
    ]

    assert "Fire Safety Sop" in titles


@pytest.mark.asyncio
async def test_retrieval_pump_vibration(
    retriever,
):
    """Pump vibration query should find the inspection report."""

    results = await retriever.retrieve_context(
        query="pump abnormal vibration",
        top_k=5,
    )

    assert len(results) > 0

    assert any(
        "Inspection Report" in result["title"]
        for result in results
    )


# -------------------------------------------------------------------
# top_k Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_top_k_limits_results(
    retriever,
):
    """Retriever should return no more than top_k results."""

    results = await retriever.retrieve_context(
        query="pump safety",
        top_k=1,
    )

    assert len(results) <= 1


@pytest.mark.asyncio
async def test_top_k_zero_returns_empty(
    retriever,
):
    """top_k=0 should return an empty list."""

    results = await retriever.retrieve_context(
        query="pump safety",
        top_k=0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_negative_top_k_returns_empty(
    retriever,
):
    """Negative top_k should return an empty list."""

    results = await retriever.retrieve_context(
        query="pump safety",
        top_k=-1,
    )

    assert results == []


# -------------------------------------------------------------------
# Empty Query Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_returns_empty(
    retriever,
):
    """Empty query should not raise an exception."""

    results = await retriever.retrieve_context(
        query="",
        top_k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_whitespace_query_returns_empty(
    retriever,
):
    """Whitespace-only query should return empty."""

    results = await retriever.retrieve_context(
        query="     ",
        top_k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_none_query_returns_empty(
    retriever,
):
    """Non-string/None query should return empty."""

    results = await retriever.retrieve_context(
        query=None,
        top_k=5,
    )

    assert results == []


# -------------------------------------------------------------------
# No Result Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unrelated_query_returns_empty(
    retriever,
):
    """Completely unrelated query should produce no results."""

    results = await retriever.retrieve_context(
        query="quantum rocket ocean astronomy",
        top_k=5,
    )

    assert results == []


# -------------------------------------------------------------------
# Result Schema Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_schema(
    retriever,
):
    """Every retrieval result must follow the expected schema."""

    results = await retriever.retrieve_context(
        query="pump",
        top_k=5,
    )

    assert len(results) > 0

    for result in results:

        assert set(result.keys()) == {
            "document_id",
            "title",
            "snippet",
            "score",
        }

        assert isinstance(
            result["document_id"],
            str,
        )

        assert isinstance(
            result["title"],
            str,
        )

        assert isinstance(
            result["snippet"],
            str,
        )

        assert isinstance(
            result["score"],
            float,
        )


# -------------------------------------------------------------------
# File ID Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_id_filter(
    knowledge_dirs,
):
    """
    file_id should restrict search to matching uploaded files.
    """

    upload_file = (
        knowledge_dirs["uploads"]
        / "file-123-pump.txt"
    )

    upload_file.write_text(
        """
        Uploaded pump inspection document.

        Pump pressure is within the acceptable range.
        Pump vibration must be monitored.
        """,
        encoding="utf-8",
    )

    backend = _LocalFilesystemSearch()
    retriever = Retriever(backend=backend)

    results = await retriever.retrieve_context(
        query="pump vibration",
        file_id="file-123",
        top_k=5,
    )

    assert len(results) > 0

    assert all(
        "file-123" in result["document_id"]
        for result in results
    )


@pytest.mark.asyncio
async def test_unknown_file_id_returns_empty(
    retriever,
):
    """Unknown file_id should return no results."""

    results = await retriever.retrieve_context(
        query="pump",
        file_id="file-does-not-exist",
        top_k=5,
    )

    assert results == []


# -------------------------------------------------------------------
# Health Check Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_when_directories_exist(
    retriever,
):
    """Retriever should report ready when local directories exist."""

    result = await retriever.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_backend_failure():
    """Backend health failure should safely return False."""

    class BrokenBackend:

        async def search(
            self,
            query,
            top_k,
            file_id=None,
        ):
            return []

        async def is_ready(self):
            raise RuntimeError(
                "Fake backend failure"
            )

    retriever = Retriever(
        backend=BrokenBackend()
    )

    result = await retriever.health_check()

    assert result is False


# -------------------------------------------------------------------
# Vector Store Contract Test
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_store_backend_contract():
    """
    Verify that a backend implementing search() and is_ready()
    satisfies the VectorStoreBackend protocol.
    """

    class FakeVectorStore:

        async def search(
            self,
            query,
            top_k,
            file_id=None,
        ):
            return [
                {
                    "document_id": "doc-001",
                    "title": "Test Document",
                    "snippet": "Pump safety information",
                    "score": 0.91,
                }
            ]

        async def is_ready(self):
            return True

    backend = FakeVectorStore()

    assert isinstance(
        backend,
        VectorStoreBackend,
    )


@pytest.mark.asyncio
async def test_retriever_with_vector_backend():
    """
    Retriever should work with a vector-store-compatible backend.
    """

    class FakeVectorStore:

        async def search(
            self,
            query,
            top_k,
            file_id=None,
        ):
            return [
                {
                    "document_id": "vector-doc-001",
                    "title": "Vector Document",
                    "snippet": "Semantic pump safety result",
                    "score": 0.97,
                }
            ]

        async def is_ready(self):
            return True

    retriever = Retriever(
        backend=FakeVectorStore()
    )

    results = await retriever.retrieve_context(
        query="pump safety",
        top_k=5,
    )

    assert len(results) == 1

    assert results[0]["document_id"] == (
        "vector-doc-001"
    )

    assert results[0]["score"] == 0.97


# -------------------------------------------------------------------
# Backend Error Test
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_error_is_propagated():
    """
    Retriever should propagate backend search errors.

    routes_chat.py handles this error gracefully at API level.
    """

    class FailingBackend:

        async def search(
            self,
            query,
            top_k,
            file_id=None,
        ):
            raise RuntimeError(
                "Vector store unavailable"
            )

        async def is_ready(self):
            return False

    retriever = Retriever(
        backend=FailingBackend()
    )

    with pytest.raises(RuntimeError):
        await retriever.retrieve_context(
            query="pump",
            top_k=5,
        )