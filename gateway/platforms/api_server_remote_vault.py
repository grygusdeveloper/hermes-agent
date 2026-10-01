"""Scoped server-side Hermes Vault API (``/v1/vault/*``) for the Obsidian Bridge remote-vault provider.

``RemoteVaultRoutesMixin`` (inherited by ``APIServerAdapter``) exposes only the configured roots under
``gateway.api_server.remote_vault.base_dir`` — never an arbitrary filesystem: paths are POSIX-relative,
hidden/``.git`` segments are refused, symlinks may not escape their root, and updates of an existing note
are base-hash checked. The surface is off unless ``remote_vault.enabled`` (or ``HERMES_REMOTE_VAULT_ENABLED``)
says otherwise; ``HERMES_REMOTE_VAULT_*`` env keys override the config block.
api_server helpers are imported lazily: the origin imports this module (a top-level import would cycle).
"""

import asyncio
import hashlib
import hmac
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors api_server's optional import
    web = None  # type: ignore[assignment]

REMOTE_VAULT_DEFAULT_BASE_DIR = "/root/workspace/hermes-vault"
REMOTE_VAULT_DEFAULT_ROOTS = (
    "Inbox", "Ideas", "Operations", "Projects", "Research", "System",
    "research", "inner-brain", "staging", "drafts",
)
REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES = 262144
REMOTE_VAULT_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".yaml", ".yml"})
REMOTE_VAULT_SEARCH_DEFAULT_LIMIT = 20
REMOTE_VAULT_SEARCH_MAX_LIMIT = 100
REMOTE_VAULT_ATTACHMENTS_DEFAULT_LIMIT = 100
REMOTE_VAULT_ATTACHMENTS_MAX_LIMIT = 500
REMOTE_VAULT_MAX_SCAN_FILES = 5000
REMOTE_VAULT_SNIPPET_CHARS = 160
_TEXT_CACHE_MAX_ENTRIES = 2048

# /v1/capabilities "endpoints" rows: name -> (method, path).
CAPABILITY_ENDPOINTS = (
    ("remote_vault_roots", ("GET", "/v1/vault/roots")),
    ("remote_vault_tree", ("GET", "/v1/vault/tree?root={root}")),
    ("remote_vault_note", ("GET", "/v1/vault/note?path={path}")),
    ("remote_vault_update", ("POST", "/v1/vault/update")),
    ("remote_vault_search", ("GET", "/v1/vault/search?q={query}&root={root}")),
    ("remote_vault_attachments", ("GET", "/v1/vault/attachments?root={root}")),
)


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [
        ("GET", "/v1/vault/roots", self._handle_vault_roots),
        ("GET", "/v1/vault/tree", self._handle_vault_tree),
        ("GET", "/v1/vault/search", self._handle_vault_search),
        ("GET", "/v1/vault/attachments", self._handle_vault_attachments),
        ("GET", "/v1/vault/note", self._handle_vault_note),
        ("POST", "/v1/vault/update", self._handle_vault_update)]


def _coerce_request_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp an untrusted integer query value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _normalize_rel(value: Any, *, allow_hidden: bool = False) -> Optional[str]:
    """Safe vault-relative POSIX path, or None (absolute, backslash, ``.``/``..``, ``.git``, hidden)."""
    raw = str(value or "").strip()
    if not raw or "\\" in raw:
        return None
    posix = PurePosixPath(raw)
    if posix.is_absolute() or not posix.parts:
        return None
    for part in posix.parts:
        if part in {"", ".", ".."} or part == ".git" or (part.startswith(".") and not allow_hidden):
            return None
    return "/".join(posix.parts)


def _coerce_roots(value: Any, *, allow_hidden: bool = False) -> tuple[str, ...]:
    """Configured roots (comma string or list) normalized and de-duplicated; unsafe entries dropped."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        items = list(REMOTE_VAULT_DEFAULT_ROOTS)
    roots = (_normalize_rel(item, allow_hidden=allow_hidden) for item in items)
    return tuple(dict.fromkeys(root for root in roots if root))


def _path_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _iso_mtime(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    except OSError:
        return ""


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_text_note(path: Path) -> bool:
    return path.suffix.lower() in REMOTE_VAULT_TEXT_SUFFIXES


def _is_skipped_name(name: str, allow_hidden: bool) -> bool:
    return name == ".git" or (name.startswith(".") and not allow_hidden)


def _snippet(text: str, match_index: int, needle_len: int) -> str:
    half = max(20, (REMOTE_VAULT_SNIPPET_CHARS - max(needle_len, 1)) // 2)
    start = max(0, match_index - half)
    end = min(len(text), match_index + max(needle_len, 1) + half)
    snippet = ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")
    snippet = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return redact_sensitive_text(snippet)[:REMOTE_VAULT_SNIPPET_CHARS]


def _result_metadata(rel: str, root: str) -> Dict[str, Any]:
    """Provenance fields the Obsidian Bridge renders for server-side (non-canonical) results."""
    return {
        "source": "hermes-remote-vault", "sourceKind": "hermes", "sourceLabel": "Hermes Remote Vault",
        "canonical": False, "badge": "🟪 Hermes", "title": PurePosixPath(rel).stem or rel,
        "displayPath": rel, "openMode": "remote-vault-mirror", "root": root,
    }


def remote_vault_settings() -> Dict[str, Any]:
    """``gateway.api_server.remote_vault`` with ``HERMES_REMOTE_VAULT_*`` env overrides.

    Filesystem exposure fails closed: an unreadable config or absent block means disabled.
    """
    from gateway.platforms.api_server import MAX_REQUEST_BYTES, _coerce_request_bool
    raw: Dict[str, Any] = {}
    try:
        from hermes_cli.config import cfg_get, load_config
        cfg = cfg_get(load_config(), "gateway", "api_server", "remote_vault", default={})
        if isinstance(cfg, dict):
            raw = cfg
    except Exception:
        raw = {}
    enabled = _coerce_request_bool(os.getenv("HERMES_REMOTE_VAULT_ENABLED"), bool(raw.get("enabled", False)))
    allow_hidden = _coerce_request_bool(
        os.getenv("HERMES_REMOTE_VAULT_ALLOW_HIDDEN"), bool(raw.get("allow_hidden", False)))
    writable = _coerce_request_bool(os.getenv("HERMES_REMOTE_VAULT_WRITABLE"), bool(raw.get("writable", True)))
    base_dir = os.getenv("HERMES_REMOTE_VAULT_BASE_DIR") or str(raw.get("base_dir") or REMOTE_VAULT_DEFAULT_BASE_DIR)
    roots = os.getenv("HERMES_REMOTE_VAULT_ROOTS") or raw.get("roots") or list(REMOTE_VAULT_DEFAULT_ROOTS)
    try:
        max_file_bytes = int(os.getenv("HERMES_REMOTE_VAULT_MAX_FILE_BYTES") or raw.get("max_file_bytes")
                             or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES)
    except (TypeError, ValueError):
        max_file_bytes = REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES
    return {
        "enabled": enabled,
        "base_dir": Path(base_dir).expanduser(),
        "roots": _coerce_roots(roots, allow_hidden=allow_hidden),
        "max_file_bytes": max(1024, min(max_file_bytes, MAX_REQUEST_BYTES)),
        "allow_hidden": allow_hidden,
        "writable": writable,
    }


def capability_features(settings: Dict[str, Any]) -> Dict[str, bool]:
    enabled = bool(settings.get("enabled"))
    return {
        "remote_vault": enabled,
        "remote_vault_update": enabled and bool(settings.get("writable", True)),
        "remote_vault_search": enabled,
        "remote_vault_attachments": enabled,
    }


def capability_limits(settings: Dict[str, Any]) -> Dict[str, int]:
    return {
        "remote_vault_search_max_results": REMOTE_VAULT_SEARCH_MAX_LIMIT,
        "remote_vault_attachments_max_results": REMOTE_VAULT_ATTACHMENTS_MAX_LIMIT,
        "remote_vault_max_scan_files": REMOTE_VAULT_MAX_SCAN_FILES,
        "remote_vault_snippet_chars": REMOTE_VAULT_SNIPPET_CHARS,
        "remote_vault_max_file_bytes": int(settings.get("max_file_bytes") or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES),
    }


def _vault_error(message: str, *, status: int = 400, code: str = "remote_vault_error") -> "web.Response":
    from gateway.platforms.api_server import _openai_error
    return web.json_response(_openai_error(message, code=code), status=status)


_DISABLED = ("Remote vault API is disabled.", 503, "remote_vault_disabled")


class RemoteVaultRoutesMixin:
    """``/v1/vault/{roots,tree,note,update,search,attachments}`` handlers and their scoped-path helpers."""

    # -- path scoping -----------------------------------------------------------------

    @staticmethod
    def _remote_vault_base(settings: Dict[str, Any]) -> Optional[Path]:
        try:
            resolved = settings["base_dir"].resolve(strict=True)
        except OSError:
            return None
        return resolved if resolved.is_dir() else None

    @staticmethod
    def _remote_vault_root_for_rel(rel: str, settings: Dict[str, Any]) -> Optional[str]:
        for root in sorted(settings.get("roots") or (), key=len, reverse=True):
            if rel == root or rel.startswith(root + "/"):
                return root
        return None

    def _remote_vault_resolve(
        self, rel: str, settings: Dict[str, Any],
    ) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
        """``(candidate, root_dir, error)``: the candidate (may not exist yet) stays inside its resolved root."""
        base = self._remote_vault_base(settings)
        if base is None:
            return None, None, "Remote vault base is unavailable."
        root = self._remote_vault_root_for_rel(rel, settings)
        if not root:
            return None, None, "Path is outside the allowed remote vault roots."
        try:
            root_dir = (base / root).resolve(strict=True)
        except OSError:
            return None, None, "Remote vault root is unavailable."
        if not _path_within(root_dir, base) or not root_dir.is_dir():
            return None, None, "Remote vault root is unavailable."
        try:
            candidate = (base / rel).resolve(strict=False)
        except OSError:
            return None, None, "Remote vault path is invalid."
        if not _path_within(candidate, root_dir):
            return None, None, "Path is outside the allowed remote vault root."
        return candidate, root_dir, None

    def _remote_vault_existing_roots(
        self, settings: Dict[str, Any], *, root_filter: Optional[str] = None,
    ) -> tuple[Optional[Path], list[tuple[str, Path]], Optional[str], int, str]:
        """``(base, [(root, root_dir)], error, status, code)`` for configured roots that exist under base."""
        base = self._remote_vault_base(settings)
        if base is None:
            return None, [], "Remote vault base is unavailable.", 503, "remote_vault_unavailable"
        roots = tuple(settings.get("roots") or ())
        if root_filter:
            normalized = _normalize_rel(root_filter, allow_hidden=bool(settings.get("allow_hidden")))
            if not normalized or normalized not in roots:
                return base, [], "Root is not allowed.", 400, "remote_vault_root_not_allowed"
            roots = (normalized,)
        resolved: list[tuple[str, Path]] = []
        for root in roots:
            root_dir, _, err = self._remote_vault_resolve(root, settings)
            if err or root_dir is None:
                if root_filter:
                    return base, [], err or "Remote vault root is unavailable.", 404, "remote_vault_root_unavailable"
                continue
            resolved.append((root, root_dir))
        return base, resolved, None, 200, ""

    def _remote_vault_walk(self, root_dir: Path, settings: Dict[str, Any]):
        """Yield ``(kind, resolved_path)`` for safe folders and files under a resolved root; hidden/.git
        entries are skipped and symlinks are followed only while they stay inside the root."""
        allow_hidden = bool(settings.get("allow_hidden"))
        for current, dirs, files in os.walk(root_dir, followlinks=False):
            cur = Path(current)
            safe_dirs: list[str] = []
            for dirname in dirs:
                if _is_skipped_name(dirname, allow_hidden):
                    continue
                try:
                    resolved = (cur / dirname).resolve(strict=True)
                except OSError:
                    continue
                if resolved.is_dir() and _path_within(resolved, root_dir):
                    safe_dirs.append(dirname)
                    yield "folder", resolved
            dirs[:] = safe_dirs
            for filename in files:
                if _is_skipped_name(filename, allow_hidden):
                    continue
                try:
                    resolved = (cur / filename).resolve(strict=True)
                except OSError:
                    continue
                if resolved.is_file() and _path_within(resolved, root_dir):
                    yield "file", resolved

    def _remote_vault_iter_files(
        self, root_dir: Path, settings: Dict[str, Any], *, want_text: Optional[bool], stats: Dict[str, Any],
    ):
        """Safe files under a root, bounded by REMOTE_VAULT_MAX_SCAN_FILES across one request (``stats``)."""
        for kind, resolved in self._remote_vault_walk(root_dir, settings):
            if kind != "file":
                continue
            stats["scanned_files"] = stats.get("scanned_files", 0) + 1
            if stats["scanned_files"] > REMOTE_VAULT_MAX_SCAN_FILES:
                stats["truncated"] = True
                return
            if want_text is None or _is_text_note(resolved) == want_text:
                yield resolved

    def _remote_vault_cached_text_hash(self, resolved: Path) -> tuple[str, str, int]:
        """``(text, sha256, size)``; unchanged notes (mtime_ns + size) are not re-read for repeat searches."""
        cache = getattr(self, "_remote_vault_text_cache", None)
        if cache is None:
            cache = self._remote_vault_text_cache = {}
        stat = resolved.stat()
        key = str(resolved)
        cached = cache.get(key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2], cached[3], stat.st_size
        data = resolved.read_bytes()
        text = data.decode("utf-8")
        sha = _hash_bytes(data)
        if len(cache) > _TEXT_CACHE_MAX_ENTRIES:
            cache.clear()
        cache[key] = (stat.st_mtime_ns, stat.st_size, text, sha)
        return text, sha, stat.st_size

    # -- blocking workers (run via asyncio.to_thread) -----------------------------------

    def _remote_vault_tree_sync(self, root_rel: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        base, roots, err, status, code = self._remote_vault_existing_roots(settings, root_filter=root_rel)
        if err or base is None or not roots:
            return {"error": err or "Remote vault root is unavailable.", "status": status, "code": code}
        _, root_dir = roots[0]
        max_file_bytes = int(settings.get("max_file_bytes") or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES)
        entries: list[Dict[str, Any]] = []
        for kind, resolved in self._remote_vault_walk(root_dir, settings):
            rel = resolved.relative_to(base).as_posix()
            if kind == "folder":
                entries.append({"path": rel, "type": "folder", "state": "server-only"})
                continue
            if not _is_text_note(resolved):
                continue
            try:
                size = resolved.stat().st_size
                if size > max_file_bytes:
                    continue
                sha = _hash_bytes(resolved.read_bytes())
            except OSError:
                continue
            entries.append({"path": rel, "type": "file", "size": size, "modified": _iso_mtime(resolved),
                            "hash": sha, "sha256": sha, "state": "server-only"})
        entries.sort(key=lambda e: (e["type"] != "folder", e["path"]))
        return {"entries": entries}

    def _remote_vault_search_sync(
        self, query: str, settings: Dict[str, Any], *, root_filter: Optional[str], limit: int,
    ) -> Dict[str, Any]:
        """Literal (never regex) case-insensitive path/content search over text notes."""
        base, roots, err, status, code = self._remote_vault_existing_roots(settings, root_filter=root_filter)
        if err or base is None:
            return {"error": err or "Remote vault base is unavailable.", "status": status, "code": code}
        needle = query.casefold()
        max_file_bytes = int(settings.get("max_file_bytes") or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES)
        matches: list[Dict[str, Any]] = []
        stats: Dict[str, Any] = {"scanned_files": 0, "truncated": False}
        for root, root_dir in roots:
            for resolved in self._remote_vault_iter_files(root_dir, settings, want_text=True, stats=stats):
                try:
                    if resolved.stat().st_size > max_file_bytes:
                        continue
                    rel = resolved.relative_to(base).as_posix()
                    path_hit = needle in rel.casefold()
                    try:
                        text, sha, size = self._remote_vault_cached_text_hash(resolved)
                    except UnicodeDecodeError:
                        continue
                except OSError:
                    continue
                idx = text.casefold().find(needle)
                if not path_hit and idx < 0:
                    continue
                matches.append({
                    **_result_metadata(rel, root), "path": rel, "type": "file",
                    "match": "content" if idx >= 0 else "path",
                    "snippet": _snippet(text, idx, len(query)) if idx >= 0 else "",
                    "size": size, "modified": _iso_mtime(resolved), "hash": sha, "sha256": sha,
                })
            if stats.get("truncated"):
                break
        matches.sort(key=lambda e: e["path"])
        return {"entries": matches[:limit], "truncated": bool(stats.get("truncated") or len(matches) > limit),
                "scanned_files": int(stats.get("scanned_files") or 0)}

    def _remote_vault_attachments_sync(
        self, settings: Dict[str, Any], *, root_filter: Optional[str], limit: int,
    ) -> Dict[str, Any]:
        """Metadata only (never content) for non-text files."""
        base, roots, err, status, code = self._remote_vault_existing_roots(settings, root_filter=root_filter)
        if err or base is None:
            return {"error": err or "Remote vault base is unavailable.", "status": status, "code": code}
        attachments: list[Dict[str, Any]] = []
        stats: Dict[str, Any] = {"scanned_files": 0, "truncated": False}
        for root, root_dir in roots:
            for resolved in self._remote_vault_iter_files(root_dir, settings, want_text=False, stats=stats):
                suffix = resolved.suffix.lower()
                if not suffix:
                    continue
                try:
                    size = resolved.stat().st_size
                except OSError:
                    continue
                rel = resolved.relative_to(base).as_posix()
                media_type, _ = mimetypes.guess_type(str(resolved))
                attachments.append({
                    **_result_metadata(rel, root), "path": rel, "type": "attachment",
                    "extension": suffix.lstrip("."), "mediaType": media_type or "application/octet-stream",
                    "size": size, "modified": _iso_mtime(resolved),
                })
            if stats.get("truncated"):
                break
        attachments.sort(key=lambda e: e["path"])
        return {"entries": attachments[:limit],
                "truncated": bool(stats.get("truncated") or len(attachments) > limit),
                "scanned_files": int(stats.get("scanned_files") or 0)}

    def _remote_vault_update_sync(
        self, rel: str, encoded: bytes, base_hash: Any, settings: Dict[str, Any],
    ) -> tuple[int, Dict[str, Any]]:
        """Atomic (tmp + rename) write; an existing note requires the caller's matching ``baseHash``."""
        path, root_dir, resolve_err = self._remote_vault_resolve(rel, settings)
        if resolve_err and "unavailable" in resolve_err:
            return 503, {"error": resolve_err, "code": "remote_vault_unavailable"}
        if resolve_err or path is None or root_dir is None:
            return 400, {"error": resolve_err or "Path is invalid.", "code": "remote_vault_invalid_path"}
        if not _is_text_note(path):
            return 415, {"error": "Remote-vault updates are limited to text/note file extensions.",
                         "code": "remote_vault_not_text"}
        try:
            parent = path.parent.resolve(strict=False)
            if not _path_within(parent, root_dir):
                return 400, {"error": "Path is outside the allowed remote vault root.",
                             "code": "remote_vault_invalid_path"}
            if path.exists():
                resolved = path.resolve(strict=True)
                if not resolved.is_file() or not _path_within(resolved, root_dir):
                    return 404, {"error": "Path is not a writable remote-vault note.", "code": "remote_vault_not_found"}
                if not isinstance(base_hash, str) or not base_hash:
                    return 428, {"error": "baseHash is required when updating an existing remote-vault note.",
                                 "code": "remote_vault_base_hash_required"}
                current_hash = _hash_bytes(resolved.read_bytes())
                if not hmac.compare_digest(current_hash.encode(), base_hash.encode()):
                    return 409, {"accepted": False,
                                 "detail": "Remote note changed since it was cached (hash mismatch).",
                                 "currentHash": current_hash, "hash": current_hash, "sha256": current_hash}
            parent.mkdir(parents=True, exist_ok=True)
            tmp = parent / f".{path.name}.hermes-tmp-{uuid.uuid4().hex}"
            try:
                tmp.write_bytes(encoded)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        except OSError:
            return 500, {"error": "Remote-vault update failed.", "code": "remote_vault_update_failed"}
        new_hash = _hash_bytes(encoded)
        return 200, {"accepted": True, "detail": "Remote update stored in Hermes Vault.", "path": rel,
                     "hash": new_hash, "sha256": new_hash, "size": len(encoded), "modified": _iso_mtime(path)}

    def _remote_vault_note_sync(self, rel: str, settings: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        path, _, err = self._remote_vault_resolve(rel, settings)
        if err or path is None:
            return 404, {"error": err or "Path is unavailable.", "code": "remote_vault_not_found"}
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_file() or not _is_text_note(resolved):
                return 404, {"error": "Path is not a readable remote-vault note.", "code": "remote_vault_not_found"}
            if resolved.stat().st_size > int(settings.get("max_file_bytes") or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES):
                return 413, {"error": "Remote-vault note exceeds the configured size limit.",
                             "code": "remote_vault_too_large"}
            data = resolved.read_bytes()
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return 415, {"error": "Remote-vault note is not valid UTF-8 text.", "code": "remote_vault_not_text"}
        except OSError:
            return 404, {"error": "Remote-vault note is unavailable.", "code": "remote_vault_not_found"}
        sha = _hash_bytes(data)
        return 200, {"path": rel, "content": content, "hash": sha, "sha256": sha, "size": len(data),
                     "modified": _iso_mtime(resolved)}

    def _remote_vault_roots_sync(self, settings: Dict[str, Any]) -> Optional[list[str]]:
        base, roots, _, _, _ = self._remote_vault_existing_roots(settings)
        return None if base is None else [root for root, _ in roots]

    # -- HTTP handlers ------------------------------------------------------------------

    def _remote_vault_gate(self, request: "web.Request") -> tuple[Optional[Dict[str, Any]], Optional["web.Response"]]:
        """Auth first (401), then the enabled switch (503); returns the request's settings snapshot."""
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        settings = remote_vault_settings()
        if not settings.get("enabled"):
            message, status, code = _DISABLED
            return None, _vault_error(message, status=status, code=code)
        return settings, None

    @staticmethod
    def _remote_vault_worker_response(result: Dict[str, Any], fallback: str) -> Optional["web.Response"]:
        if "error" not in result:
            return None
        return _vault_error(str(result.get("error") or fallback), status=int(result.get("status") or 400),
                            code=str(result.get("code") or "remote_vault_error"))

    async def _handle_vault_roots(self, request: "web.Request") -> "web.Response":
        """GET /v1/vault/roots — configured roots that exist under the vault base."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        roots = await asyncio.to_thread(self._remote_vault_roots_sync, settings)
        if roots is None:
            return _vault_error("Remote vault base is unavailable.", status=503, code="remote_vault_unavailable")
        return web.json_response({"object": "list", "roots": roots, "data": roots})

    async def _handle_vault_tree(self, request: "web.Request") -> "web.Response":
        """GET /v1/vault/tree?root=... — recursive listing of folders and size-bounded text notes."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        root_rel = _normalize_rel(request.query.get("root"), allow_hidden=bool(settings.get("allow_hidden")))
        if not root_rel or root_rel not in (settings.get("roots") or ()):
            return _vault_error("Root is not allowed.", status=400, code="remote_vault_root_not_allowed")
        result = await asyncio.to_thread(self._remote_vault_tree_sync, root_rel, settings)
        if (failure := self._remote_vault_worker_response(result, "Remote vault tree failed.")) is not None:
            return failure
        entries = result.get("entries") or []
        return web.json_response({"object": "list", "root": root_rel, "entries": entries, "data": entries})

    async def _handle_vault_search(self, request: "web.Request") -> "web.Response":
        """GET /v1/vault/search?q=...&root=...&limit=... — literal search over scoped text notes."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        query = str(request.query.get("q") or "").strip()
        if len(query) < 2:
            return _vault_error("Search query must contain at least 2 characters.", status=400,
                                code="remote_vault_query_too_short")
        if len(query) > 128:
            return _vault_error("Search query is too long.", status=400, code="remote_vault_query_too_long")
        root_filter = request.query.get("root") or None
        limit = _coerce_request_int(request.query.get("limit"), REMOTE_VAULT_SEARCH_DEFAULT_LIMIT, 1,
                                    REMOTE_VAULT_SEARCH_MAX_LIMIT)
        result = await asyncio.to_thread(
            self._remote_vault_search_sync, query, settings, root_filter=root_filter, limit=limit)
        if (failure := self._remote_vault_worker_response(result, "Remote vault search failed.")) is not None:
            return failure
        entries = result.get("entries") or []
        return web.json_response({
            "object": "list", "query": query, "root": root_filter, "limit": limit,
            "truncated": bool(result.get("truncated")), "scannedFiles": int(result.get("scanned_files") or 0),
            "entries": entries, "data": entries,
        })

    async def _handle_vault_attachments(self, request: "web.Request") -> "web.Response":
        """GET /v1/vault/attachments?root=...&limit=... — metadata-only listing of scoped non-text files."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        root_filter = request.query.get("root") or None
        limit = _coerce_request_int(request.query.get("limit"), REMOTE_VAULT_ATTACHMENTS_DEFAULT_LIMIT, 1,
                                    REMOTE_VAULT_ATTACHMENTS_MAX_LIMIT)
        result = await asyncio.to_thread(
            self._remote_vault_attachments_sync, settings, root_filter=root_filter, limit=limit)
        if (failure := self._remote_vault_worker_response(
                result, "Remote vault attachments listing failed.")) is not None:
            return failure
        entries = result.get("entries") or []
        return web.json_response({
            "object": "list", "root": root_filter, "limit": limit,
            "truncated": bool(result.get("truncated")), "scannedFiles": int(result.get("scanned_files") or 0),
            "entries": entries, "data": entries,
        })

    async def _handle_vault_note(self, request: "web.Request") -> "web.Response":
        """GET /v1/vault/note?path=... — one size-bounded UTF-8 text note with its sha256."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        rel = _normalize_rel(request.query.get("path"), allow_hidden=bool(settings.get("allow_hidden")))
        if not rel:
            return _vault_error("Path is invalid.", status=400, code="remote_vault_invalid_path")
        status, body = await asyncio.to_thread(self._remote_vault_note_sync, rel, settings)
        if "error" in body:
            return _vault_error(body["error"], status=status, code=body["code"])
        return web.json_response(body, status=status)

    async def _handle_vault_update(self, request: "web.Request") -> "web.Response":
        """POST /v1/vault/update {path, content, baseHash} — base-hash-checked atomic text update."""
        settings, err = self._remote_vault_gate(request)
        if err:
            return err
        if not settings.get("writable"):
            return _vault_error("Remote vault updates are disabled.", status=403, code="remote_vault_read_only")
        body, err_resp = await self._read_json_body(request)
        if err_resp:
            return err_resp
        rel = _normalize_rel(body.get("path"), allow_hidden=bool(settings.get("allow_hidden")))
        content = body.get("content")
        if not rel or not isinstance(content, str):
            return _vault_error("Request must include safe path and string content.", status=400,
                                code="remote_vault_invalid_update")
        encoded = content.encode("utf-8")
        if len(encoded) > int(settings.get("max_file_bytes") or REMOTE_VAULT_DEFAULT_MAX_FILE_BYTES):
            return _vault_error("Remote-vault note exceeds the configured size limit.", status=413,
                                code="remote_vault_too_large")
        base_hash = body.get("baseHash") or body.get("hash") or body.get("sha256")
        status, result = await asyncio.to_thread(self._remote_vault_update_sync, rel, encoded, base_hash, settings)
        if "error" in result:
            return _vault_error(result["error"], status=status, code=result["code"])
        return web.json_response(result, status=status)
