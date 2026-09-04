"""
tests/test_routes_agent.py

Unit tests for backend/api/routes_agent.py

These tests isolate the Agent API from:
- real database
- real LLM/model router
- real knowledge-base retriever
- real executor/tools

Only the route/controller behavior is tested.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import routes_agent


# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------


class FakeConversation:
    def __init__(self, conversation_id: str):
        self.id = conversation_id


class FakeDB:
    """Dummy DB object used only by mocked CRUD functions."""

    pass


class FakePlanner:
    """Fake planner so no real model/planner is called."""

    async def create_plan(
        self,
        message,
        history=None,
        context_chunks=None,
        has_attachment=False,
    ):
        return {
            "goal": message,
            "task_type": "general",
            "steps": [],
        }


class FakeExecutionResult:
    def __init__(self):
        self.success = True
        self.results = []
        self.outputs = {}


class FakeExecutor:
    """Fake executor so no real tools are executed."""

    async def execute(self, plan):
        return FakeExecutionResult()


class FakeRetriever:
    """Fake retriever so no real vector store/files are accessed."""

    async def retrieve_context(
        self,
        query,
        file_id=None,
        top_k=5,
    ):
        return []


# ---------------------------------------------------------------------------
# Fake CRUD
# ---------------------------------------------------------------------------


class FakeCRUD:
    def __init__(self):
        self.messages = []

    async def get_conversation(
        self,
        db,
        conversation_id,
    ):
        if conversation_id == "missing":
            return None

        return FakeConversation(conversation_id)

    async def get_conversation_history(
        self,
        db,
        conversation_id,
    ):
        return []

    async def create_conversation(self, db):
        return FakeConversation("new-conversation")

    async def add_message(
        self,
        db,
        conversation_id,
        role,
        content,
        metadata=None,
    ):
        self.messages.append(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata,
            }
        )

        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_crud(monkeypatch):
    """
    Replace real CRUD calls with in-memory fake CRUD.
    """

    crud = FakeCRUD()

    monkeypatch.setattr(
        routes_agent.crud,
        "get_conversation",
        crud.get_conversation,
    )

    monkeypatch.setattr(
        routes_agent.crud,
        "get_conversation_history",
        crud.get_conversation_history,
    )

    monkeypatch.setattr(
        routes_agent.crud,
        "create_conversation",
        crud.create_conversation,
    )

    monkeypatch.setattr(
        routes_agent.crud,
        "add_message",
        crud.add_message,
    )

    return crud


@pytest.fixture
def client(fake_crud):
    """
    Create an isolated FastAPI application containing only
    the agent router.
    """

    # Runtime task storage is process-local.
    routes_agent._TASKS.clear()

    app = FastAPI()
    app.include_router(routes_agent.router)

    # ------------------------------------------------------------------
    # Fake DB dependency
    # ------------------------------------------------------------------

    async def fake_get_db():
        yield FakeDB()

    app.dependency_overrides[
        routes_agent.get_db
    ] = fake_get_db

    # ------------------------------------------------------------------
    # Fake Agent dependencies
    # ------------------------------------------------------------------

    app.dependency_overrides[
        routes_agent.get_planner
    ] = lambda: FakePlanner()

    app.dependency_overrides[
        routes_agent.get_executor
    ] = lambda: FakeExecutor()

    app.dependency_overrides[
        routes_agent.get_retriever
    ] = lambda: FakeRetriever()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    routes_agent._TASKS.clear()


# ===========================================================================
# Route registration tests
# ===========================================================================





# ===========================================================================
# POST /api/agent/run
# ===========================================================================


def test_agent_run_creates_new_conversation(client):
    """
    If conversation_id is omitted, a new conversation must be created.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Hello agent",
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert data["task_id"]
    assert data["conversation_id"] == "new-conversation"
    assert data["status"] == "queued"
    assert data["message"]


def test_agent_run_existing_conversation(client):
    """
    Existing conversation_id must be accepted.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Continue my task",
            "conversation_id": "conv-123",
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert data["task_id"]
    assert data["conversation_id"] == "conv-123"
    assert data["status"] == "queued"


def test_agent_run_missing_conversation(client):
    """
    Unknown conversation_id must return 404.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Test task",
            "conversation_id": "missing",
        },
    )

    assert response.status_code == 404


def test_agent_run_empty_message(client):
    """
    Empty message must fail Pydantic validation.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "",
        },
    )

    assert response.status_code == 422


def test_agent_run_whitespace_message(client):
    """
    Whitespace-only message must fail validation.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "     ",
        },
    )

    assert response.status_code == 422


def test_agent_run_missing_message(client):
    """
    message is required.
    """

    response = client.post(
        "/api/agent/run",
        json={},
    )

    assert response.status_code == 422


def test_agent_run_unknown_field_rejected(client):
    """
    AgentRunRequest uses extra='forbid'.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Hello",
            "unknown_field": "abc",
        },
    )

    assert response.status_code == 422


def test_agent_run_file_id(client):
    """
    file_id must be accepted by the route.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Summarize this document",
            "file_id": "file-123",
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert data["task_id"]
    assert data["conversation_id"] == "new-conversation"
    assert data["status"] == "queued"


def test_agent_run_top_k_default(client):
    """
    Request should work without explicitly supplying top_k.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Test",
        },
    )

    assert response.status_code == 202


def test_agent_run_top_k_minimum_validation(client):
    """
    top_k must be >= 1.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Test",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_agent_run_top_k_maximum_validation(client):
    """
    top_k must be <= 20.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Test",
            "top_k": 21,
        },
    )

    assert response.status_code == 422


def test_agent_run_top_k_valid_value(client):
    """
    Valid top_k must be accepted.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Test",
            "top_k": 10,
        },
    )

    assert response.status_code == 202


# ===========================================================================
# GET /api/agent/status/{task_id}
# ===========================================================================


def test_agent_status_unknown_task(client):
    """
    Unknown task_id must return 404.
    """

    response = client.get(
        "/api/agent/status/does-not-exist",
    )

    assert response.status_code == 404


def test_agent_status_after_run(client):
    """
    A task returned by POST /run must be queryable through
    GET /status/{task_id}.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Run a test task",
        },
    )

    assert response.status_code == 202

    task_id = response.json()["task_id"]

    assert task_id

    # Give the background task a small amount of time to run.
    final_data = None

    for _ in range(20):
        status_response = client.get(
            f"/api/agent/status/{task_id}",
        )

        assert status_response.status_code == 200

        final_data = status_response.json()

        if final_data["status"] in {
            "completed",
            "failed",
        }:
            break

        time.sleep(0.01)

    assert final_data is not None

    assert final_data["task_id"] == task_id
    assert final_data["conversation_id"] == "new-conversation"
    assert final_data["message"] == "Run a test task"

    assert final_data["status"] in {
        "queued",
        "planning",
        "executing",
        "completed",
        "failed",
    }


def test_agent_status_response_structure(client):
    """
    Status endpoint must return all expected response fields.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Check response structure",
        },
    )

    assert response.status_code == 202

    task_id = response.json()["task_id"]

    response = client.get(
        f"/api/agent/status/{task_id}",
    )

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "task_id",
        "conversation_id",
        "status",
        "message",
        "task_type",
        "goal",
        "current_step",
        "current_tool",
        "plan",
        "results",
        "outputs",
        "error",
        "created_at",
        "updated_at",
        "completed_at",
    }

    assert required_fields.issubset(data.keys())


def test_agent_status_contains_plan_after_background_execution(client):
    """
    Once the fake planner runs, the generated plan should appear
    in task status.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Create a plan",
        },
    )

    assert response.status_code == 202

    task_id = response.json()["task_id"]

    plan_found = False

    for _ in range(30):
        response = client.get(
            f"/api/agent/status/{task_id}",
        )

        assert response.status_code == 200

        data = response.json()

        if data["plan"] is not None:
            plan_found = True

            assert data["plan"]["goal"] == "Create a plan"
            assert data["plan"]["task_type"] == "general"
            assert data["plan"]["steps"] == []

            break

        time.sleep(0.01)

    assert plan_found


# ===========================================================================
# Task ID tests
# ===========================================================================


def test_agent_run_generates_unique_task_ids(client):
    """
    Every submitted task must receive a different task_id.
    """

    response1 = client.post(
        "/api/agent/run",
        json={
            "message": "Task one",
        },
    )

    response2 = client.post(
        "/api/agent/run",
        json={
            "message": "Task two",
        },
    )

    assert response1.status_code == 202
    assert response2.status_code == 202

    task_id_1 = response1.json()["task_id"]
    task_id_2 = response2.json()["task_id"]

    assert task_id_1
    assert task_id_2
    assert task_id_1 != task_id_2


# ===========================================================================
# CRUD persistence test
# ===========================================================================


def test_agent_run_persists_user_message(client, fake_crud):
    """
    Agent submission should persist the user's message through CRUD.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Persist this message",
        },
    )

    assert response.status_code == 202

    assert len(fake_crud.messages) == 1

    message = fake_crud.messages[0]

    assert message["conversation_id"] == "new-conversation"
    assert message["role"] == "user"
    assert message["content"] == "Persist this message"


# ===========================================================================
# Final state test
# ===========================================================================


def test_agent_task_reaches_completed_state_with_fake_executor(client):
    """
    With a fake planner producing zero steps and a fake executor
    returning success, the background task should complete.
    """

    response = client.post(
        "/api/agent/run",
        json={
            "message": "Complete this test",
        },
    )

    assert response.status_code == 202

    task_id = response.json()["task_id"]

    completed = False
    final_data = None

    for _ in range(30):
        response = client.get(
            f"/api/agent/status/{task_id}",
        )

        assert response.status_code == 200

        final_data = response.json()

        if final_data["status"] == "completed":
            completed = True
            break

        if final_data["status"] == "failed":
            break

        time.sleep(0.01)

    assert completed, final_data

    assert final_data["error"] is None
    assert final_data["completed_at"] is not None
    assert final_data["task_id"] == task_id
def test_routes_agent_agent_components_are_compatible():
    """
    Verify that routes_agent.py is wired against the actual interfaces
    exposed by planner.py, executer.py and memory.py.

    This test does not start the FastAPI server and does not call
    any real LLM, database, retriever or external service.
    """

    import inspect

    from backend.agent.planner import Planner
    from backend.agent.executer import Executor
    from backend.agent.memory import AgentMemory

    # ---------------------------------------------------------------
    # 1. Planner compatibility
    # ---------------------------------------------------------------

    assert hasattr(Planner, "create_plan")
    assert callable(Planner.create_plan)

    planner_signature = inspect.signature(
        Planner.create_plan
    )

    planner_parameters = planner_signature.parameters

    assert "message" in planner_parameters
    assert "history" in planner_parameters
    assert "context_chunks" in planner_parameters
    assert "has_attachment" in planner_parameters

    # ---------------------------------------------------------------
    # 2. Executor compatibility
    # ---------------------------------------------------------------

    assert hasattr(Executor, "execute")
    assert callable(Executor.execute)

    executor_signature = inspect.signature(
        Executor.execute
    )

    executor_parameters = executor_signature.parameters

    assert "plan" in executor_parameters

    assert hasattr(Executor, "register_tool")
    assert callable(Executor.register_tool)

    assert hasattr(Executor, "available_tools")
    assert callable(Executor.available_tools)

    # ---------------------------------------------------------------
    # 3. Memory compatibility
    # ---------------------------------------------------------------

    required_memory_methods = [
        "add_message",
        "set_context",
        "get_context",
        "set_task",
        "record_step",
        "get_step",
        "get_step_output",
        "get_steps",
        "get_outputs",
    ]

    for method_name in required_memory_methods:
        assert hasattr(
            AgentMemory,
            method_name,
        ), f"AgentMemory missing method: {method_name}"

        assert callable(
            getattr(AgentMemory, method_name)
        )

    # ---------------------------------------------------------------
    # 4. Verify actual Memory behavior
    # ---------------------------------------------------------------

    memory = AgentMemory(
        task_id="compatibility-test",
        conversation_id="conversation-test",
    )

    memory.add_message(
        role="user",
        content="Compatibility test",
    )

    assert len(memory.get_messages()) == 1

    memory.set_context(
        "file_id",
        "test-file",
    )

    assert (
        memory.get_context("file_id")
        == "test-file"
    )

    memory.set_task(
        goal="Compatibility test",
        task_type="general",
    )

    assert memory.goal == "Compatibility test"
    assert memory.task_type == "general"

    # ---------------------------------------------------------------
    # 5. Verify Memory step interface used by Executor results
    # ---------------------------------------------------------------

    memory.record_step(
        step_id="step_1",
        tool="test_tool",
        success=True,
        output="test output",
        error=None,
    )

    step = memory.get_step("step_1")

    assert step is not None

    assert (
        memory.get_step_output("step_1")
        == "test output"
    )

    outputs = memory.get_outputs()

    assert isinstance(outputs, dict)

    # ---------------------------------------------------------------
    # 6. Verify Executor can accept a tool and expose it
    # ---------------------------------------------------------------

    executor = Executor()

    def test_tool():
        return "tool output"

    executor.register_tool(
        "test_tool",
        test_tool,
    )

    assert "test_tool" in executor.available_tools()

    # ---------------------------------------------------------------
    # 7. Verify routes_agent imports the same real classes
    # ---------------------------------------------------------------

    from backend.api import routes_agent

    assert routes_agent.Planner is Planner
    assert routes_agent.Executor is Executor
    assert routes_agent.AgentMemory is AgentMemory   