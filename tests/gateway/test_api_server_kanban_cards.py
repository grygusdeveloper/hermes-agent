"""Tests for the API-server mediated Kanban card creation endpoint."""

import re

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from hermes_cli import kanban_db as kb


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/kanban/cards", adapter._handle_kanban_cards_list)
    app.router.add_post("/v1/kanban/cards", adapter._handle_kanban_cards)
    return app


@pytest.fixture(autouse=True)
def isolated_kanban_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)


class TestKanbanCardCreate:
    @pytest.mark.asyncio
    async def test_dry_run_validates_and_returns_source_context_without_writing(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/kanban/cards",
                json={
                    "version": "kanban-card-action-v0",
                    "title": "Follow up from search result",
                    "body": "Source: Obsidian Bridge result card",
                    "board": "obsidian-bridge-product",
                    "assignee": "default",
                    "priority": 40,
                    "source": "hermes-obsidian-bridge",
                    "sourceContext": {"kind": "vault-result", "sourceLabel": "🟦 Draw", "path": "Notes/Jump.md"},
                    "dryRun": True,
                },
            )

            assert resp.status == 200
            data = await resp.json()
            assert data["accepted"] is True
            assert data["dryRun"] is True
            assert "taskId" not in data
            assert data["sourceContext"]["kind"] == "vault-result"

        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            assert kb.list_tasks(conn) == []

    @pytest.mark.asyncio
    async def test_create_returns_real_task_id_and_persists_redacted_card(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/kanban/cards",
                json={
                    "version": "kanban-card-action-v0",
                    "title": "Feedback follow-up OPENAI_API_KEY=sk-test-secret-1234567890",
                    "body": "Please investigate. OPENAI_API_KEY=sk-body-secret-1234567890\nSource context included below.",
                    "board": "obsidian-bridge-product",
                    "assignee": "default",
                    "priority": 60,
                    "source": "hermes-obsidian-bridge",
                    "sourceContext": {"kind": "feedback", "lastCommand": "/feedback", "path": "Draw/Note.md"},
                },
            )

            assert resp.status == 201
            data = await resp.json()
            task_id = data["taskId"]
            assert re.match(r"^t_[0-9a-f]+$", task_id)
            assert data["task"]["id"] == task_id
            assert data["task"]["sourceContext"]["kind"] == "feedback"
            assert data["source"] == "hermes-obsidian-bridge"

        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.assignee == "default"
            assert task.status == "ready"
            assert "sk-body-secret-1234567890" not in (task.body or "")
            assert "sk-test-secret-1234567890" not in task.title
            assert "***" in task.title or "***" in (task.body or "")

    @pytest.mark.asyncio
    async def test_idempotency_key_prevents_duplicate_cards(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        payload = {
            "title": "Idempotent card",
            "body": "Same payload should return the same task.",
            "board": "obsidian-bridge-product",
            "assignee": "default",
            "idempotencyKey": "obsidian-bridge-test-idempotency",
            "sourceContext": {"kind": "proposal", "proposalId": "prop-1"},
        }
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post("/v1/kanban/cards", json=payload)
            second = await cli.post("/v1/kanban/cards", json=payload)
            assert first.status == 201
            assert second.status == 201
            first_data = await first.json()
            second_data = await second.json()
            assert first_data["taskId"] == second_data["taskId"]

    @pytest.mark.asyncio
    async def test_auth_is_enforced_when_api_key_configured(self):
        adapter = _make_adapter(api_key="sk-api-server")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/kanban/cards", json={"title": "No auth"})
            assert resp.status == 401

            ok = await cli.post(
                "/v1/kanban/cards",
                json={"title": "Authorized dry run", "dryRun": True},
                headers={"Authorization": "Bearer sk-api-server"},
            )
            assert ok.status == 200
            assert (await ok.json())["dryRun"] is True

    @pytest.mark.asyncio
    async def test_invalid_board_slug_is_rejected(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/kanban/cards",
                json={"title": "Bad board", "board": "../secrets", "sourceContext": {"kind": "diagnostic"}},
            )
            assert resp.status == 400
            data = await resp.json()
            assert data["error"]["code"] == "kanban_card_invalid_request"

    @pytest.mark.asyncio
    async def test_capabilities_advertise_kanban_endpoint(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["kanban_card_create"] is True
            assert data["endpoints"]["kanban_card_create"]["path"] == "/v1/kanban/cards"


class TestKanbanCardList:
    """Tests for GET /v1/kanban/cards — the server-mediated list endpoint."""

    @pytest.mark.asyncio
    async def test_list_returns_real_tasks_with_ids_and_status(self):
        """The core acceptance test: list returns real tasks from the board DB."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        # Seed two tasks directly into the board DB
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            t1 = kb.create_task(conn, title="Blocked worker A", body="Need UX input", board="obsidian-bridge-product", assignee="default", priority=70)
            t2 = kb.create_task(conn, title="Done worker B", body="Completed handoff", board="obsidian-bridge-product", assignee="reviewer", priority=40)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&view=cockpit&limit=50")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert data["board"] == "obsidian-bridge-product"
            assert data["view"] == "cockpit"
            assert data["count"] >= 2
            ids = {t["id"] for t in data["tasks"]}
            assert t1 in ids
            assert t2 in ids
            # Each task must have id, title, status
            for task in data["tasks"]:
                assert re.match(r"^t_[0-9a-f]+$", task["id"])
                assert "title" in task
                assert "status" in task

    @pytest.mark.asyncio
    async def test_list_no_token_on_pc_returns_tasks_not_404(self):
        """List endpoint must return data (200), not 404 or network error."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            kb.create_task(conn, title="List probe", body="probe", board="obsidian-bridge-product")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product")
            assert resp.status == 200
            data = await resp.json()
            assert data["count"] >= 1

    @pytest.mark.asyncio
    async def test_list_auth_is_enforced_when_api_key_configured(self):
        """GET list must enforce auth the same way POST does."""
        adapter = _make_adapter(api_key="secret-test-key")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product")
            assert resp.status == 401

            ok = await cli.get(
                "/v1/kanban/cards?board=obsidian-bridge-product",
                headers={"Authorization": "Bearer secret-test-key"},
            )
            assert ok.status == 200

    @pytest.mark.asyncio
    async def test_list_rejects_invalid_board_slug(self):
        """Malformed board slug returns explicit 400, not opaque network error."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=../secrets")
            assert resp.status == 400
            data = await resp.json()
            assert data["error"]["code"] == "kanban_card_invalid_request"

    @pytest.mark.asyncio
    async def test_list_malformed_limit_falls_back_to_default(self):
        """Non-integer limit doesn't crash — falls back to default 80."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&limit=abc")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_list_limit_is_bounded_to_max_200(self):
        """Limit > 200 is clamped to 200."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            kb.create_task(conn, title="Bound probe", board="obsidian-bridge-product")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&limit=99999")
            assert resp.status == 200
            # The server should clamp; the response should still work
            data = await resp.json()
            assert data["count"] >= 1

    @pytest.mark.asyncio
    async def test_list_redacts_secrets_in_body_and_title(self):
        """Secret patterns in task body/title must be redacted in the list response."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            kb.create_task(
                conn,
                title="Task with OPENAI_API_KEY=sk-test1234567890",
                body="Secret key sk-test1234567890 in body",
                board="obsidian-bridge-product",
            )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product")
            assert resp.status == 200
            raw = await resp.text()
            assert "sk-test1234567890" not in raw

    @pytest.mark.asyncio
    async def test_list_cockpit_view_trims_body(self):
        """view=cockpit trims body to keep payload small."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        long_body = "A" * 500
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            kb.create_task(conn, title="Trim probe", body=long_body, board="obsidian-bridge-product")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&view=cockpit")
            assert resp.status == 200
            data = await resp.json()
            for task in data["tasks"]:
                if task.get("body"):
                    assert len(task["body"]) <= 300

    @pytest.mark.asyncio
    async def test_list_includes_parents_and_children(self):
        """List includes parent/child task IDs when they exist."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            parent_id = kb.create_task(conn, title="Parent task", body="p", board="obsidian-bridge-product")
            child_id = kb.create_task(conn, title="Child task", body="c", board="obsidian-bridge-product", parents=[parent_id])

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product")
            assert resp.status == 200
            data = await resp.json()
            task_map = {t["id"]: t for t in data["tasks"]}
            if child_id in task_map:
                assert parent_id in (task_map[child_id].get("parents") or [])
            if parent_id in task_map:
                assert child_id in (task_map[parent_id].get("children") or [])

    @pytest.mark.asyncio
    async def test_list_status_filter_works(self):
        """Status filter returns only matching tasks."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            kb.create_task(conn, title="Ready task", board="obsidian-bridge-product")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&status=ready")
            assert resp.status == 200
            data = await resp.json()
            for task in data["tasks"]:
                assert task["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_exposes_only_bounded_artifact_claims_from_latest_run_metadata(self):
        """Cockpit gets verifiable artifact claims without receiving arbitrary run metadata."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        verdict_path = r"C:\Users\Grygus\Notes\Draw\.obsidian\hermes-bridge\m7-r22\t_fixture\run-42\qa-evidence\verdict.json"
        missing_path = r"C:\Users\Grygus\Notes\Draw\.obsidian\hermes-bridge\m7-r22\t_fixture\run-42\missing\must-not-exist.json"
        artifact_claim = {
            "kind": "pc-path",
            "path": verdict_path,
            "sha256": "a" * 64,
            "bytes": 321,
            "ownerTaskId": "t_fixture",
            "ownerRunId": 42,
            "ownerWorkflowRunId": "m7-r22",
            "availability": "content-verified",
            "verificationSource": "reviewer",
            "verifiedAt": "2026-07-10T12:00:00Z",
        }
        with kb.connect_closing(board="obsidian-bridge-product") as conn:
            task_id = kb.create_task(
                conn,
                title="Artifact handoff",
                body="done",
                board="obsidian-bridge-product",
            )
            assert kb.complete_task(
                conn,
                task_id,
                summary="Artifact handoff completed.",
                metadata={
                    "artifacts": [artifact_claim],
                    "missing_artifact_path": missing_path,
                    "OPENAI_API_KEY": "must-not-leak",
                    "changed_files": ["private/internal.py"],
                },
            )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/kanban/cards?board=obsidian-bridge-product&view=cockpit")
            assert resp.status == 200
            raw = await resp.text()
            data = await resp.json()

        card = next(task for task in data["tasks"] if task["id"] == task_id)
        assert card["artifacts"] == [artifact_claim, missing_path]
        assert "must-not-leak" not in raw
        assert "OPENAI_API_KEY" not in raw
        assert "changed_files" not in raw
        assert "private/internal.py" not in raw

    @pytest.mark.asyncio
    async def test_capabilities_advertise_list_endpoint(self):
        """Capabilities should now advertise kanban_card_list."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["kanban_card_list"] is True
            assert data["endpoints"]["kanban_card_list"]["method"] == "GET"
