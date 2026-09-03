"""
tests/test_api.py

FastAPI integration tests for the Sovereign AI Workbench.

Tests:
- Root endpoint
- Basic health endpoint
- Chat health endpoint
- Chat success flow
- Chat validation
- Conversation handling
- RAG/retriever flow
- Model failure handling
- Database failure handling

Real LLM, vector database and production database are NOT used.
Dependencies are mocked so the tests remain fast and offline.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.routes_chat import (
    get_model_router,
    get_retriever,
)
from backend.database.init_db import get_db


# -------------------------------------------------------------------
# Fake dependencies
# -------------------------------------------------------------------

class FakeModelRouter:
    """Fake local model router."""

    async def generate(
        self,
        message,
        task_type,
        history,
        context_chunks=None,
    ):
        return {
            "answer": f"Test response for: {message}",
            "model": "fake-model",
        }

    async def health_check(self):
        return True


class FakeRetriever:
    """Fake knowledge-base retriever."""

    async def retrieve_context(
        self,
        query,
        file_id=None,
        top_k=5,
    ):
        return [
            {
                "document_id": "doc-test-001",
                "title": "Test Document",
                "snippet": "This is test knowledge.",
                "score": 0.95,
            }
        ]

    async def health_check(self):
        return True


class FakeDatabase:
    """Dummy database object."""

    pass


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def fake_model_router():
    return FakeModelRouter()


@pytest.fixture
def fake_retriever():
    return FakeRetriever()


@pytest.fixture
def client(fake_model_router, fake_retriever, monkeypatch):
    """
    Create TestClient with mocked dependencies.
    """

    async def override_get_db():
        yield FakeDatabase()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_model_router] = (
        lambda: fake_model_router
    )
    app.dependency_overrides[get_retriever] = (
        lambda: fake_retriever
    )

    # Mock database CRUD functions used by routes_chat.
    async def fake_create_conversation(db):
        return SimpleNamespace(id="test-conversation-001")

    async def fake_get_conversation(db, conversation_id):
        if conversation_id == "missing-conversation":
            return None

        return SimpleNamespace(id=conversation_id)

    async def fake_get_conversation_history(
        db,
        conversation_id,
    ):
        return []

    async def fake_add_message(
        db,
        conversation_id,
        role,
        content,
        metadata=None,
    ):
        return None

    async def fake_ping(db):
        return True

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.create_conversation",
        fake_create_conversation,
    )

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.get_conversation",
        fake_get_conversation,
    )

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.get_conversation_history",
        fake_get_conversation_history,
    )

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.add_message",
        fake_add_message,
    )

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.ping",
        fake_ping,
    )

    # Prevent real task-classifier/model execution.
    async def fake_classify_task(
        message,
        has_attachment=False,
    ):
        return "chat"

    monkeypatch.setattr(
        "backend.api.routes_chat.classify_task",
        fake_classify_task,
    )

    yield TestClient(app)

    app.dependency_overrides.clear()


# -------------------------------------------------------------------
# Root API
# -------------------------------------------------------------------

def test_root_endpoint(client):
    """GET / should return service information."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Sovereign AI Workbench"
    assert data["status"] == "running"
    assert "version" in data


# -------------------------------------------------------------------
# Basic Health
# -------------------------------------------------------------------

def test_basic_health(client):
    """GET /health should confirm API process is alive."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


# -------------------------------------------------------------------
# Chat Health
# -------------------------------------------------------------------

def test_chat_health(client):
    """GET /api/chat/health should report all dependencies ready."""

    response = client.get("/api/chat/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_router_ready"] is True
    assert data["knowledge_base_ready"] is True
    assert data["database_ready"] is True
    assert "checked_at" in data


# -------------------------------------------------------------------
# Chat Success
# -------------------------------------------------------------------

def test_chat_success(client):
    """POST /api/chat should successfully process a message."""

    response = client.post(
        "/api/chat",
        json={
            "message": "Hello Workbench"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Test response for: Hello Workbench"
    assert data["model"] == "fake-model"
    assert data["task_type"] == "chat"
    assert data["conversation_id"] == "test-conversation-001"

    assert isinstance(data["sources"], list)
    assert "created_at" in data


# -------------------------------------------------------------------
# Chat Validation
# -------------------------------------------------------------------

def test_chat_empty_message(client):
    """Empty message should return HTTP 400."""

    response = client.post(
        "/api/chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "`message` must not be empty." in data["detail"]


def test_chat_whitespace_message(client):
    """Whitespace-only message should return HTTP 400."""

    response = client.post(
        "/api/chat",
        json={
            "message": "     "
        },
    )

    assert response.status_code == 400


def test_chat_missing_message(client):
    """Missing required message field should return validation error."""

    response = client.post(
        "/api/chat",
        json={},
    )

    assert response.status_code == 422


def test_chat_invalid_json_type(client):
    """Message must be a string."""

    response = client.post(
        "/api/chat",
        json={
            "message": 12345
        },
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# Conversation Tests
# -------------------------------------------------------------------

def test_chat_with_existing_conversation(client):
    """Existing conversation ID should be accepted."""

    response = client.post(
        "/api/chat",
        json={
            "message": "Continue our conversation",
            "conversation_id": "existing-conversation-001",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["conversation_id"] == "existing-conversation-001"


def test_chat_missing_conversation(client):
    """Unknown conversation should return HTTP 404."""

    response = client.post(
        "/api/chat",
        json={
            "message": "Continue conversation",
            "conversation_id": "missing-conversation",
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert "not found" in data["detail"].lower()


# -------------------------------------------------------------------
# File / RAG Tests
# -------------------------------------------------------------------

def test_chat_with_file_id(client):
    """
    Supplying file_id should trigger knowledge-base retrieval.
    """

    response = client.post(
        "/api/chat",
        json={
            "message": "Summarize this document",
            "file_id": "file-test-001",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]
    assert len(data["sources"]) == 1

    source = data["sources"][0]

    assert source["document_id"] == "doc-test-001"
    assert source["title"] == "Test Document"
    assert source["snippet"] == "This is test knowledge."
    assert source["score"] == 0.95


# -------------------------------------------------------------------
# Model Failure
# -------------------------------------------------------------------

def test_chat_model_failure(client, monkeypatch):
    """Model failure should return HTTP 502."""

    class FailingModelRouter:

        async def generate(
            self,
            message,
            task_type,
            history,
            context_chunks=None,
        ):
            raise RuntimeError("Fake model failure")

        async def health_check(self):
            return False

    app.dependency_overrides[get_model_router] = (
        lambda: FailingModelRouter()
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "Test model failure"
        },
    )

    assert response.status_code == 502

    data = response.json()

    assert (
        "local model backend failed"
        in data["detail"].lower()
    )


# -------------------------------------------------------------------
# Database Failure
# -------------------------------------------------------------------

def test_chat_database_failure(client, monkeypatch):
    """Conversation storage failure should return HTTP 500."""

    async def failing_create_conversation(db):
        raise RuntimeError("Fake database failure")

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.create_conversation",
        failing_create_conversation,
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "Test database failure"
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert (
        "conversation storage"
        in data["detail"].lower()
    )


# -------------------------------------------------------------------
# Health Degraded Tests
# -------------------------------------------------------------------

def test_chat_health_degraded_model(client):
    """Health should become degraded if model router is unavailable."""

    class UnhealthyModelRouter:

        async def health_check(self):
            return False

    app.dependency_overrides[get_model_router] = (
        lambda: UnhealthyModelRouter()
    )

    response = client.get("/api/chat/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"
    assert data["model_router_ready"] is False


def test_chat_health_degraded_database(client, monkeypatch):
    """Health should become degraded if database ping fails."""

    async def failing_ping(db):
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(
        "backend.api.routes_chat.crud.ping",
        failing_ping,
    )

    response = client.get("/api/chat/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"
    assert data["database_ready"] is False


# -------------------------------------------------------------------
# HTTP Method Tests
# -------------------------------------------------------------------

def test_chat_get_not_allowed(client):
    """GET /api/chat should not be allowed."""

    response = client.get("/api/chat")

    assert response.status_code == 405


def test_health_post_not_allowed(client):
    """POST /health should not be allowed."""

    response = client.post("/health")

    assert response.status_code == 405


# -------------------------------------------------------------------
# Unknown Route
# -------------------------------------------------------------------

def test_unknown_endpoint(client):
    """Unknown endpoint should return 404."""

    response = client.get(
        "/api/this-endpoint-does-not-exist"
    )

    assert response.status_code == 404