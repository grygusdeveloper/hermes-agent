"""Scoped server-side Hermes Vault API (/v1/vault/*) used by the Obsidian Bridge remote-vault provider."""

import hashlib
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_VAULT_ENV = ("HERMES_REMOTE_VAULT_ENABLED", "HERMES_REMOTE_VAULT_BASE_DIR", "HERMES_REMOTE_VAULT_ROOTS",
              "HERMES_REMOTE_VAULT_WRITABLE", "HERMES_REMOTE_VAULT_ALLOW_HIDDEN", "HERMES_REMOTE_VAULT_MAX_FILE_BYTES")


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Register the vault + capabilities rows exactly as connect() does, from the adapter's route table."""
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/vault/") or path == "/v1/capabilities":
            app.router.add_route(method, path, handler)
    return app


@pytest.fixture(autouse=True)
def _no_vault_config(monkeypatch):
    for name in _VAULT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})


def _configure(monkeypatch, tmp_path, *, writable=True):
    base = tmp_path / "hermes-vault"
    (base / "staging" / "drafts").mkdir(parents=True)
    (base / "staging" / "drafts" / "note.md").write_text("# Remote\nbody\n", encoding="utf-8")
    (base / "staging" / "skip.bin").write_bytes(b"binary")
    (base / "outside").mkdir()
    monkeypatch.setenv("HERMES_REMOTE_VAULT_BASE_DIR", str(base))
    monkeypatch.setenv("HERMES_REMOTE_VAULT_ENABLED", "true")
    monkeypatch.setenv("HERMES_REMOTE_VAULT_ROOTS", "staging,research")
    monkeypatch.setenv("HERMES_REMOTE_VAULT_WRITABLE", "true" if writable else "false")
    return base


@pytest.mark.asyncio
async def test_roots_tree_and_note_are_scoped(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        roots = await cli.get("/v1/vault/roots")
        assert roots.status == 200
        # "research" is configured but absent on disk.
        assert (await roots.json())["roots"] == ["staging"]

        tree = await cli.get("/v1/vault/tree?root=staging")
        assert tree.status == 200
        paths = {entry["path"] for entry in (await tree.json())["entries"]}
        assert {"staging/drafts", "staging/drafts/note.md"} <= paths
        assert "staging/skip.bin" not in paths

        note = await cli.get("/v1/vault/note?path=staging/drafts/note.md")
        assert note.status == 200
        body = await note.json()
        assert body["content"] == "# Remote\nbody\n"
        assert body["sha256"] == hashlib.sha256((base / "staging" / "drafts" / "note.md").read_bytes()).hexdigest()

        assert (await cli.get("/v1/vault/note?path=staging/../outside/x.md")).status in {400, 404}
        assert (await cli.get("/v1/vault/tree?root=outside")).status == 400


@pytest.mark.asyncio
async def test_update_is_base_hash_checked(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    note_path = base / "staging" / "drafts" / "note.md"
    original_hash = hashlib.sha256(note_path.read_bytes()).hexdigest()
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        ok = await cli.post("/v1/vault/update", json={
            "path": "staging/drafts/note.md", "content": "# Changed\n", "baseHash": original_hash})
        assert ok.status == 200
        ok_body = await ok.json()
        assert ok_body["accepted"] is True
        assert ok_body["sha256"] == hashlib.sha256(b"# Changed\n").hexdigest()
        assert note_path.read_text(encoding="utf-8") == "# Changed\n"

        stale = await cli.post("/v1/vault/update", json={
            "path": "staging/drafts/note.md", "content": "# Clobber\n", "baseHash": original_hash})
        assert stale.status == 409
        stale_body = await stale.json()
        assert stale_body["accepted"] is False
        assert "hash mismatch" in stale_body["detail"]
        assert stale_body["currentHash"] == ok_body["sha256"]
        assert note_path.read_text(encoding="utf-8") == "# Changed\n"

        created = await cli.post("/v1/vault/update", json={"path": "staging/new/idea.md", "content": "new"})
        assert created.status == 200
        assert (base / "staging" / "new" / "idea.md").read_text(encoding="utf-8") == "new"
        # Atomic write leaves no temp files behind.
        assert not [p for p in (base / "staging").rglob(".*hermes-tmp*")]


@pytest.mark.asyncio
async def test_existing_update_requires_base_hash(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        resp = await cli.post("/v1/vault/update", json={"path": "staging/drafts/note.md",
                                                       "content": "# Blind overwrite\n"})
        assert resp.status == 428
    assert (base / "staging" / "drafts" / "note.md").read_text(encoding="utf-8") == "# Remote\nbody\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(("path", "status"), [
    ("outside/x.md", 400), ("staging/.hidden.md", 400), ("/etc/passwd.md", 400), ("staging/x.sh", 415),
])
async def test_update_refuses_paths_outside_scoped_text_notes(monkeypatch, tmp_path, path, status):
    base = _configure(monkeypatch, tmp_path)
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        resp = await cli.post("/v1/vault/update", json={"path": path, "content": "x"})
        assert resp.status == status
    assert not (base / "outside" / "x.md").exists()
    assert not (base / "staging" / ".hidden.md").exists()


@pytest.mark.asyncio
async def test_update_can_be_disabled(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, writable=False)
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        resp = await cli.post("/v1/vault/update", json={"path": "staging/x.md", "content": "x"})
        assert resp.status == 403


@pytest.mark.asyncio
async def test_auth_is_enforced_on_vault_endpoints(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        assert (await cli.get("/v1/vault/roots")).status == 401
        assert (await cli.get("/v1/vault/roots", headers={"Authorization": "Bearer wrong"})).status == 401
        assert (await cli.post("/v1/vault/update", json={"path": "staging/x.md", "content": "x"})).status == 401
        ok = await cli.get("/v1/vault/roots", headers={"Authorization": "Bearer sk-secret"})
        assert ok.status == 200


@pytest.mark.asyncio
async def test_symlink_escape_is_blocked(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    outside_note = base / "outside" / "secret.md"
    outside_note.write_text("secret", encoding="utf-8")
    os.symlink(outside_note, base / "staging" / "link.md")
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        tree = await cli.get("/v1/vault/tree?root=staging")
        assert tree.status == 200
        assert "staging/link.md" not in {entry["path"] for entry in (await tree.json())["entries"]}
        assert (await cli.get("/v1/vault/note?path=staging/link.md")).status == 404
        write = await cli.post("/v1/vault/update", json={"path": "staging/link.md", "content": "pwn"})
        assert write.status in {400, 404}
    assert outside_note.read_text(encoding="utf-8") == "secret"


@pytest.mark.asyncio
async def test_vault_is_off_by_default_and_can_be_disabled(monkeypatch, tmp_path):
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        # No config block and no env: filesystem exposure fails closed.
        assert (await cli.get("/v1/vault/roots")).status == 503
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_REMOTE_VAULT_ENABLED", "false")
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        assert (await cli.get("/v1/vault/roots")).status == 503
        assert (await cli.get("/v1/vault/search?q=remote")).status == 503
        assert (await cli.get("/v1/vault/attachments")).status == 503


@pytest.mark.asyncio
async def test_config_block_enables_vault_without_env(monkeypatch, tmp_path):
    base = tmp_path / "vault"
    (base / "Inbox").mkdir(parents=True)
    (base / "Inbox" / "today.md").write_text("hello", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"gateway": {"api_server": {"remote_vault": {
        "enabled": True, "base_dir": str(base), "roots": ["Inbox"], "writable": False}}}})
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        assert (await (await cli.get("/v1/vault/roots")).json())["roots"] == ["Inbox"]
        assert (await cli.post("/v1/vault/update", json={"path": "Inbox/today.md", "content": "x"})).status == 403
        caps = await (await cli.get("/v1/capabilities")).json()
    assert caps["features"]["remote_vault"] is True
    assert caps["features"]["remote_vault_update"] is False
    assert caps["endpoints"]["remote_vault_update"] == {"method": "POST", "path": "/v1/vault/update"}
    assert caps["limits"]["remote_vault_search_max_results"] == 100


@pytest.mark.asyncio
async def test_search_is_literal_scoped_and_bounded(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    fake_key = "sk-" + "test-secret-value-0123456789abcdef"  # assembled so the fixture is not a literal key
    (base / "staging" / "drafts" / "tokens.json").write_text(
        '{"note":"alpha rocket", "api_key":"%s"}' % fake_key, encoding="utf-8")
    (base / "staging" / ".hidden.md").write_text("alpha hidden", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        assert (await cli.get("/v1/vault/search?q=a")).status == 400

        literal = await cli.get("/v1/vault/search?q=.*")
        assert literal.status == 200
        assert (await literal.json())["entries"] == []

        resp = await cli.get("/v1/vault/search?q=alpha&root=staging&limit=99999")
        assert resp.status == 200
        body = await resp.json()
        assert body["limit"] == 100
        paths = {entry["path"] for entry in body["entries"]}
        assert "staging/drafts/tokens.json" in paths
        assert "staging/.hidden.md" not in paths
        hit = next(entry for entry in body["entries"] if entry["path"].endswith("tokens.json"))
        assert len(hit["snippet"]) <= 160
        assert fake_key not in hit["snippet"]
        assert hit["match"] == "content"
        assert (hit["sourceKind"], hit["sourceLabel"], hit["canonical"], hit["badge"], hit["openMode"],
                hit["title"]) == ("hermes", "Hermes Remote Vault", False, "🟪 Hermes", "remote-vault-mirror", "tokens")

        assert (await cli.get("/v1/vault/search?q=alpha&root=../outside")).status == 400


@pytest.mark.asyncio
async def test_search_blocks_symlink_escapes(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    outside_note = base / "outside" / "secret.md"
    outside_note.write_text("escape-needle", encoding="utf-8")
    os.symlink(outside_note, base / "staging" / "escaped.md")
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        resp = await cli.get("/v1/vault/search?q=escape-needle&root=staging")
        assert resp.status == 200
        assert (await resp.json())["entries"] == []


@pytest.mark.asyncio
async def test_attachments_are_metadata_only_and_scoped(monkeypatch, tmp_path):
    base = _configure(monkeypatch, tmp_path)
    (base / "staging" / "image.png").write_bytes(b"\x89PNG\r\n")
    (base / "staging" / ".hidden.png").write_bytes(b"hidden")
    outside_file = base / "outside" / "leak.png"
    outside_file.write_bytes(b"leak")
    os.symlink(outside_file, base / "staging" / "leak.png")
    async with TestClient(TestServer(_create_app(_make_adapter()))) as cli:
        resp = await cli.get("/v1/vault/attachments?root=staging&limit=abc")
        assert resp.status == 200
        body = await resp.json()
        assert body["limit"] == 100
        paths = {entry["path"] for entry in body["entries"]}
        assert {"staging/image.png", "staging/skip.bin"} <= paths
        assert not paths & {"staging/drafts/note.md", "staging/.hidden.png", "staging/leak.png"}
        image = next(entry for entry in body["entries"] if entry["path"] == "staging/image.png")
        assert image["mediaType"] == "image/png"
        assert (image["sourceKind"], image["canonical"], image["openMode"]) == ("hermes", False, "remote-vault-mirror")
        assert "content" not in image

        assert (await cli.get("/v1/vault/attachments?root=../outside")).status == 400
