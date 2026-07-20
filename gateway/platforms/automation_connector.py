"""Fail-closed Main-side automation service for one Todoist sandbox action.

The only supported write is ``Create task``.  Its reversal is explicitly a
*compensating deletion*: deleting the created task does not restore Todoist
history and is never represented as undo.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

ACTION_VERSION = "hermes-automation-action-v2"
CAPABILITY_VERSION = "todoist-create-compensating-v1"
CONNECTOR = "todoist"
ACTION = "Create task"
REVERSAL_DESCRIPTION = (
    "Task deletion is a compensating deletion and does not restore Todoist history."
)
MAX_PREVIEW_TTL = 300
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,255}$")
_PLACEHOLDER_FRAGMENTS = (
    "placeholder", "pending", "unknown", "missing", "none", "null",
    "tbd", "n-a", "not-applicable", "todo", "example",
)


class AutomationError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class AmbiguousProviderError(Exception):
    """The provider request began but its result could not be observed."""


class TodoistProvider(Protocol):
    async def get_project(self, project_id: str) -> Mapping[str, Any]: ...
    async def create_task(self, payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]: ...
    async def find_tasks(self, project_id: str, marker: str) -> list[Mapping[str, Any]]: ...
    async def delete_task(self, task_id: str, request_id: str) -> None: ...
    async def task_exists(self, task_id: str, project_id: str) -> bool: ...


class AiohttpTodoistProvider:
    """Minimal real provider. Credentials remain in this Main-side object."""
    def __init__(self, token: str, base_url: str = "https://api.todoist.com/api/v1"):
        self._token, self._base_url = token, base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import aiohttp
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self._token}"
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, self._base_url + path, headers=headers, **kwargs) as response:
                    if response.status == 404:
                        raise AutomationError(409, "provider_target_missing", "Configured Todoist target is unavailable.")
                    if response.status >= 400:
                        raise AutomationError(502, "provider_rejected", "Todoist rejected the redacted automation request.")
                    if response.status == 204:
                        return None
                    return await response.json()
        except AutomationError:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as exc:
            if method in {"POST", "DELETE"}:
                raise AmbiguousProviderError() from exc
            raise AutomationError(503, "provider_unavailable", "Todoist is temporarily unavailable.") from exc

    async def get_project(self, project_id: str) -> Mapping[str, Any]:
        return await self._request("GET", f"/projects/{project_id}")

    async def create_task(self, payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        return await self._request("POST", "/tasks", json=dict(payload), headers={"X-Request-Id": request_id})

    async def find_tasks(self, project_id: str, marker: str) -> list[Mapping[str, Any]]:
        matches: list[Mapping[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):
            params: dict[str, Any] = {"project_id": project_id, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            page = await self._request("GET", "/tasks", params=params)
            tasks = page.get("results", []) if isinstance(page, Mapping) else page
            matches.extend(t for t in tasks if marker in str(t.get("description") or ""))
            cursor = page.get("next_cursor") if isinstance(page, Mapping) else None
            if not cursor:
                break
        return matches

    async def delete_task(self, task_id: str, request_id: str) -> None:
        await self._request("DELETE", f"/tasks/{task_id}", headers={"X-Request-Id": request_id})

    async def task_exists(self, task_id: str, project_id: str) -> bool:
        try:
            task = await self._request("GET", f"/tasks/{task_id}")
            # Todoist API v1 may 404 a removed task OR return it carrying an
            # is_deleted tombstone; treat either as factual absence so a
            # compensating deletion reconciles honestly instead of stalling.
            if task.get("is_deleted"):
                return False
            return str(task.get("project_id")) == project_id
        except AutomationError as exc:
            if exc.code == "provider_target_missing":
                return False
            raise


@dataclass(frozen=True)
class AutomationConfig:
    token: str
    project_id: str
    account_alias: str
    tenant: str
    db_path: str
    preview_ttl: int = MAX_PREVIEW_TTL

    @classmethod
    def from_env(cls) -> "AutomationConfig":
        home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
        try:
            ttl = min(MAX_PREVIEW_TTL, max(1, int(os.getenv("TODOIST_AUTOMATION_PREVIEW_TTL", "300"))))
        except ValueError:
            ttl = MAX_PREVIEW_TTL
        return cls(
            token=os.getenv("TODOIST_API_TOKEN", "").strip(),
            project_id=os.getenv("TODOIST_AUTOMATION_PROJECT_ID", "").strip(),
            account_alias=os.getenv("TODOIST_AUTOMATION_ACCOUNT_ALIAS", "personal-inbox-test-lane").strip(),
            tenant=os.getenv("TODOIST_AUTOMATION_TENANT", "default").strip(),
            db_path=os.getenv("HERMES_AUTOMATION_DB_PATH", str(home / "automation_actions.db")),
            preview_ttl=ttl,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.project_id and self.account_alias and self.tenant)


class AutomationService:
    def __init__(self, config: Optional[AutomationConfig] = None, provider: Optional[TodoistProvider] = None):
        self.config = config or AutomationConfig.from_env()
        self.provider = provider or (AiohttpTodoistProvider(self.config.token) if self.config.enabled else None)
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if self.config.enabled:
            self._open_db()

    def _open_db(self) -> None:
        db = Path(self.config.db_path).expanduser()
        db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS previews (
          preview_id TEXT PRIMARY KEY, account_alias TEXT NOT NULL, tenant TEXT NOT NULL,
          connector TEXT NOT NULL, action TEXT NOT NULL, target_id TEXT NOT NULL,
          fields_json TEXT NOT NULL, permission_snapshot TEXT NOT NULL,
          target_version TEXT NOT NULL, request_digest TEXT NOT NULL,
          actor TEXT NOT NULL, rule_id TEXT NOT NULL, event_id TEXT NOT NULL, source TEXT NOT NULL,
          idempotency_key TEXT NOT NULL,
          created_at REAL NOT NULL, expires_at REAL NOT NULL, consumed_at REAL
        );
        CREATE TABLE IF NOT EXISTS actions (
          action_id TEXT PRIMARY KEY, preview_id TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
          binding_digest TEXT NOT NULL, state TEXT NOT NULL, marker TEXT NOT NULL,
          external_id TEXT, request_id TEXT NOT NULL, reversal_action_id TEXT UNIQUE NOT NULL,
          reversal_state TEXT, reversal_request_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at REAL NOT NULL,
          transition TEXT NOT NULL, state TEXT NOT NULL, connector TEXT NOT NULL,
          action TEXT NOT NULL, account_alias TEXT NOT NULL, tenant TEXT NOT NULL,
          preview_id TEXT, action_id TEXT, external_id TEXT, reversal_action_id TEXT,
          request_digest TEXT, target_version TEXT
        );
        """)
        os.chmod(db, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db) + suffix)
            if sidecar.exists():
                os.chmod(sidecar, 0o600)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def capability(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {"version": ACTION_VERSION, "enabled": False, "actions": []}
        return {
            "route": "/v1/automation/actions",
            "version": ACTION_VERSION,
            "enabled": True,
            "actions": [{
                "connectorId": CONNECTOR, "action": ACTION,
                "accountAlias": self.config.account_alias, "tenant": self.config.tenant,
                "capabilityVersion": CAPABILITY_VERSION, "reversalClass": "compensating",
                "previewSupported": True, "executable": True,
                # Todoist exposes no provider-isolated sandbox, so there is no
                # sandbox to "verify". Never advertise sandboxVerified=true: a
                # client that gates on it must be able to fail closed. Clients
                # gate instead on the explicit controlled-live disposable
                # test-lane proof below.
                "sandboxVerified": False,
                "providerSandbox": False,
                "testLaneClass": "controlled-live-personal-target",
                "testLaneProof": self._test_lane_proof(),
                "capabilityExpiresAt": self._iso(time.time() + min(300, self.config.preview_ttl)),
                "targetVersionRequired": True, "redactedAuditRequired": True,
                "explicitConfirmationRequired": True, "statusReconciliationSupported": True,
                "singleUsePreview": True, "compensatingReversalSupported": True,
                "reversalDescription": REVERSAL_DESCRIPTION,
            }],
        }

    def _test_lane_proof(self) -> dict[str, Any]:
        """Exact, honest proof of the controlled-live disposable test lane.

        Todoist provides no provider-isolated sandbox, so nothing here is
        "sandbox verified". The client instead gates on these concrete,
        checkable facts: writes hit live Todoist data inside one preconfigured
        personal test project, every created task is conspicuously labeled, and
        the only disposal path is a compensating deletion that does NOT restore
        Todoist history.
        """
        return {
            "providerSandbox": False,
            "isolation": "none-provider-side",
            "writesLiveProviderData": True,
            "scope": "single-preconfigured-project",
            "accountAlias": self.config.account_alias,
            "tenant": self.config.tenant,
            "targetProjectId": self.config.project_id,
            "labelPrefix": "[Hermes M8.1 test]",
            "disposal": "compensating-delete",
            "disposalRestoresHistory": False,
            "verificationMethod": "controlled-live-create-then-compensating-delete",
        }

    @staticmethod
    def _iso(epoch: float) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def _digest(cls, value: Any) -> str:
        return hashlib.sha256(cls._canonical(value).encode()).hexdigest()

    @staticmethod
    def _factual_id(value: Any, field: str) -> str:
        text = str(value or "").strip()
        lowered = text.lower()
        if (not _ID_RE.fullmatch(text)
                or any(fragment in lowered for fragment in _PLACEHOLDER_FRAGMENTS)):
            raise AutomationError(400, "invalid_factual_id", f"{field} must be a factual identifier.")
        return text

    @staticmethod
    def _bounded(value: Any, field: str, maximum: int, *, required: bool = True) -> str:
        text = str(value or "").strip()
        if (required and not text) or len(text) > maximum or any(ord(c) < 32 and c not in "\n\t" for c in text):
            raise AutomationError(400, "invalid_draft", f"{field} is missing or outside its allowed bound.")
        return text

    @classmethod
    def _target_version(cls, project: Mapping[str, Any]) -> str:
        stable = {k: project.get(k) for k in (
            "id", "name", "is_deleted", "is_archived", "is_frozen", "can_write",
            "role", "access", "updated_at",
        )}
        return cls._digest(stable)

    def _validate_project(self, project: Mapping[str, Any]) -> str:
        if str(project.get("id") or "") != self.config.project_id:
            raise AutomationError(409, "target_drift", "Configured Todoist target changed.")
        role = str(project.get("role") or "").upper()
        if (project.get("is_deleted") or project.get("is_archived") or project.get("is_frozen")
                or project.get("can_write") is False or role in {"READ_ONLY", "GUEST"}):
            raise AutomationError(409, "target_not_writable", "Configured Todoist target is not writable.")
        return self._target_version(project)

    def _audit(self, transition: str, state: str, **values: Any) -> None:
        assert self._conn
        self._conn.execute(
            """INSERT INTO audit(occurred_at,transition,state,connector,action,account_alias,tenant,
               preview_id,action_id,external_id,reversal_action_id,request_digest,target_version)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), transition, state, CONNECTOR, ACTION, self.config.account_alias,
             self.config.tenant, values.get("preview_id"), values.get("action_id"),
             values.get("external_id"), values.get("reversal_action_id"),
             values.get("request_digest"), values.get("target_version")),
        )

    async def dispatch(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if not self.config.enabled or not self.provider or not self._conn:
            raise AutomationError(503, "automation_disabled", "Automation capability is not configured.")
        if body.get("version") != ACTION_VERSION:
            raise AutomationError(400, "unsupported_version", "Unsupported automation action version.")
        operation = body.get("operation")
        if operation == "preview":
            return 200, await self.preview(body)
        if operation == "execute":
            return await self.execute(body)
        if operation == "status":
            return await self.status(body)
        if operation == "reverse":
            return await self.reverse(body)
        raise AutomationError(400, "unsupported_operation", "Unsupported automation operation.")

    async def preview(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if set(body) != {"version", "operation", "draft", "accountAlias", "tenant", "capabilityVersion", "capabilityExpiresAt"}:
            raise AutomationError(400, "invalid_preview_shape", "Preview accepts only the reviewed v2 capability and bounded draft fields.")
        if body.get("capabilityVersion") != CAPABILITY_VERSION:
            raise AutomationError(409, "capability_drift", "Automation capability version changed.")
        try:
            capability_expiry = str(body.get("capabilityExpiresAt") or "")
            from datetime import datetime
            expiry_epoch = datetime.fromisoformat(capability_expiry.replace("Z", "+00:00")).timestamp()
        except Exception as exc:
            raise AutomationError(410, "capability_expired", "Automation capability is missing or expired.") from exc
        if expiry_epoch <= time.time() or expiry_epoch > time.time() + MAX_PREVIEW_TTL + 5:
            raise AutomationError(410, "capability_expired", "Automation capability is missing or expired.")
        if body.get("accountAlias") != self.config.account_alias or body.get("tenant") != self.config.tenant:
            raise AutomationError(409, "account_drift", "Automation account or tenant changed.")
        draft = body.get("draft") if isinstance(body.get("draft"), Mapping) else {}
        allowed_draft = {"connectorId", "action", "title", "sourceKind", "sourceBadge", "sourceLabel", "notePath", "url", "projectHint", "idempotencyKey", "eventId", "actor", "ruleId"}
        if set(draft) - allowed_draft:
            raise AutomationError(400, "unexpected_draft_field", "Automation draft contains unsupported fields.")
        if draft.get("connectorId") != CONNECTOR or draft.get("action") != ACTION:
            raise AutomationError(400, "unsupported_action", "Only Todoist Create task is supported.")
        raw_title = self._bounded(draft.get("title"), "title", 180)
        normalized_fields = {
            "title": ("[Hermes M8.1 test] " + raw_title)[:200],
            "description": "Created from a controlled Hermes Obsidian Bridge test lane. Source content was withheld.",
        }
        provenance = {
            "actor": self._bounded(draft.get("actor"), "actor", 32),
            "ruleId": self._bounded(draft.get("ruleId"), "ruleId", 80),
            "eventId": self._factual_id(draft.get("eventId"), "eventId"),
            "source": self._bounded(draft.get("sourceKind"), "sourceKind", 32),
        }
        idempotency_key = self._factual_id(draft.get("idempotencyKey"), "idempotencyKey")
        normalized = {
            "connector": CONNECTOR, "action": ACTION,
            "accountAlias": self.config.account_alias, "tenant": self.config.tenant,
            "target": {"projectId": self.config.project_id}, "fields": normalized_fields,
            "idempotencyKey": idempotency_key, **provenance,
        }
        project = await self.provider.get_project(self.config.project_id)
        target_version = self._validate_project(project)
        permission_snapshot = self._canonical({
            "role": project.get("role"), "access": project.get("access"),
            "can_write": project.get("can_write"), "is_frozen": project.get("is_frozen"),
        })
        digest = self._digest(normalized)
        preview_id = "pv_" + secrets.token_hex(24)
        now, expires = time.time(), time.time() + self.config.preview_ttl
        async with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """INSERT INTO previews(
                       preview_id,account_alias,tenant,connector,action,target_id,fields_json,
                       permission_snapshot,target_version,request_digest,actor,rule_id,event_id,
                       source,idempotency_key,created_at,expires_at,consumed_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
                    (preview_id, self.config.account_alias, self.config.tenant, CONNECTOR, ACTION,
                     self.config.project_id, self._canonical(normalized_fields), permission_snapshot,
                     target_version, digest, provenance["actor"], provenance["ruleId"],
                     provenance["eventId"], provenance["source"], idempotency_key, now, expires),
                )
                self._audit("preview_minted", "previewed", preview_id=preview_id,
                            request_digest=digest, target_version=target_version)
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise AutomationError(503, "audit_unavailable", "Mandatory automation audit append failed.")
        return {
            "version": ACTION_VERSION, "operation": "preview", "state": "previewed",
            "previewId": preview_id, "expiresAt": self._iso(expires), "requestDigest": digest,
            "targetVersion": target_version, "connectorId": CONNECTOR, "action": ACTION,
            "summary": f"Create one clearly labeled Todoist test task in account {self.config.account_alias}.",
            "accountAlias": self.config.account_alias, "tenant": self.config.tenant,
            "capabilityVersion": CAPABILITY_VERSION,
            "reversalClass": "compensating", "reversalDescription": REVERSAL_DESCRIPTION,
        }

    def _preview_row(self, preview_id: str) -> sqlite3.Row:
        assert self._conn
        row = self._conn.execute("SELECT * FROM previews WHERE preview_id=?", (preview_id,)).fetchone()
        if not row:
            raise AutomationError(404, "preview_not_found", "Automation preview was not found.")
        return row

    def _action_response(self, row: Mapping[str, Any], operation: str = "execute") -> dict[str, Any]:
        preview = self._preview_row(str(row["preview_id"]))
        reversal_state = row["reversal_state"]
        state = reversal_state if operation in {"reverse", "status"} and reversal_state else row["state"]
        if state == "succeeded":
            summary = f"Created exactly one Todoist test task with factual external ID {row['external_id']}."
        elif state == "reversed":
            summary = f"Todoist task {row['external_id']} is absent after compensating deletion; Todoist history was not restored."
        elif state in {"unknown", "reversal_unknown", "executing", "reversing"}:
            summary = "Provider outcome is unknown or in flight. Do not retry blindly; use factual status reconciliation."
        else:
            summary = f"Automation action is factually {state}."
        return {
            "version": ACTION_VERSION, "operation": operation, "state": state,
            "previewId": row["preview_id"], "actionId": row["action_id"],
            "externalId": row["external_id"], "idempotencyKey": row["idempotency_key"],
            "connectorId": CONNECTOR, "action": ACTION, "summary": summary,
            "accountAlias": preview["account_alias"], "tenant": preview["tenant"],
            "capabilityVersion": CAPABILITY_VERSION, "targetVersion": preview["target_version"],
            "requestDigest": preview["request_digest"],
            "reversalClass": "compensating", "reversalDescription": REVERSAL_DESCRIPTION,
            "reversalActionId": row["reversal_action_id"], "reversalState": reversal_state,
        }

    async def execute(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if set(body) != {"version", "operation", "previewId", "idempotencyKey"}:
            raise AutomationError(400, "invalid_execute_shape", "Execute accepts only version, operation, previewId, and idempotencyKey.")
        preview_id = self._factual_id(body.get("previewId"), "previewId")
        idem = self._factual_id(body.get("idempotencyKey"), "idempotencyKey")
        binding = self._digest({"previewId": preview_id, "idempotencyKey": idem})
        async with self._lock:
            existing = self._conn.execute("SELECT * FROM actions WHERE idempotency_key=?", (idem,)).fetchone()
            if existing:
                if existing["binding_digest"] != binding:
                    raise AutomationError(409, "idempotency_conflict", "Idempotency key is bound to a different request.")
                return (202 if existing["state"] in {"executing", "unknown"} else 200), self._action_response(existing)
            preview = self._preview_row(preview_id)
            if preview["consumed_at"] is not None:
                raise AutomationError(409, "preview_replayed", "Automation preview was already consumed.")
            if time.time() >= preview["expires_at"]:
                raise AutomationError(410, "preview_expired", "Automation preview expired.")
            if preview["account_alias"] != self.config.account_alias or preview["tenant"] != self.config.tenant:
                raise AutomationError(409, "account_drift", "Automation account or tenant changed.")
            if preview["target_id"] != self.config.project_id:
                raise AutomationError(409, "target_drift", "Automation target changed.")
            if preview["connector"] != CONNECTOR or preview["action"] != ACTION or not preview["permission_snapshot"]:
                raise AutomationError(409, "preview_binding_drift", "Automation preview binding changed.")
            if idem != preview["idempotency_key"]:
                raise AutomationError(409, "idempotency_conflict", "Idempotency key does not match the opaque preview binding.")
            persisted_normalized = {
                "connector": CONNECTOR, "action": ACTION,
                "accountAlias": preview["account_alias"], "tenant": preview["tenant"],
                "target": {"projectId": preview["target_id"]},
                "fields": json.loads(preview["fields_json"]),
                "idempotencyKey": preview["idempotency_key"],
                "actor": preview["actor"], "ruleId": preview["rule_id"],
                "eventId": preview["event_id"], "source": preview["source"],
            }
            if self._digest(persisted_normalized) != preview["request_digest"]:
                raise AutomationError(409, "preview_digest_drift", "Automation preview digest changed.")
            project = await self.provider.get_project(self.config.project_id)
            if self._validate_project(project) != preview["target_version"]:
                raise AutomationError(409, "target_version_drift", "Todoist target version changed after preview.")
            action_id = "act_" + uuid.uuid4().hex
            reversal_action_id = "rev_" + uuid.uuid5(uuid.NAMESPACE_URL, action_id + ":compensating-delete").hex
            marker = "[hermes-action:" + action_id + "]"
            request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, binding))
            now = time.time()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("UPDATE previews SET consumed_at=? WHERE preview_id=? AND consumed_at IS NULL", (now, preview_id))
                self._conn.execute(
                    "INSERT INTO actions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (action_id, preview_id, idem, binding, "executing", marker, None,
                     request_id, reversal_action_id, None, None, now, now),
                )
                self._audit("execute_authorized", "executing", preview_id=preview_id,
                            action_id=action_id, request_digest=preview["request_digest"],
                            target_version=preview["target_version"])
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise AutomationError(503, "audit_unavailable", "Mandatory automation audit append failed.")
            fields = json.loads(preview["fields_json"])
            payload = {"content": fields["title"], "project_id": self.config.project_id,
                       "description": (fields.get("description", "") + "\n\n" + marker).strip()}
            try:
                created = await self.provider.create_task(payload, request_id)
            except AutomationError as exc:
                # A factual provider rejection response is definitive. It is
                # the only post-call path that can safely be classified failed.
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    self._conn.execute("UPDATE actions SET state='failed',updated_at=? WHERE action_id=?", (time.time(), action_id))
                    self._audit("execute_failed", "failed", preview_id=preview_id, action_id=action_id,
                                request_digest=preview["request_digest"], target_version=preview["target_version"])
                    self._conn.execute("COMMIT")
                except Exception:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                raise exc
            except Exception:
                # The provider call began but no definitive response was
                # observed. Never retry this POST blindly; marker-based status
                # reconciliation is the only route forward.
                external_id, new_state, status = None, "unknown", 202
            else:
                try:
                    external_id = self._factual_id(created.get("id"), "externalId")
                    new_state, status = "succeeded", 201
                except (AutomationError, AttributeError):
                    # Todoist may have created the task even when its response
                    # is malformed or lacks an ID. Treat this as ambiguous and
                    # reconcile by the unique action marker.
                    external_id, new_state, status = None, "unknown", 202
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("UPDATE actions SET state=?,external_id=?,updated_at=? WHERE action_id=?",
                                   (new_state, external_id, time.time(), action_id))
                self._audit("execute_result", new_state, preview_id=preview_id, action_id=action_id,
                            external_id=external_id, request_digest=preview["request_digest"],
                            target_version=preview["target_version"])
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                self._conn.execute("UPDATE actions SET state='unknown',external_id=NULL,updated_at=? WHERE action_id=?", (time.time(), action_id))
                status = 202
            row = self._conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return status, self._action_response(row)

    async def status(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if set(body) != {"version", "operation", "previewId", "idempotencyKey"}:
            raise AutomationError(400, "invalid_status_shape", "Status accepts only version, operation, previewId, and idempotencyKey.")
        preview_id = self._factual_id(body.get("previewId"), "previewId")
        idem = self._factual_id(body.get("idempotencyKey"), "idempotencyKey")
        async with self._lock:
            row = self._conn.execute("SELECT * FROM actions WHERE idempotency_key=?", (idem,)).fetchone()
            if not row or row["preview_id"] != preview_id:
                raise AutomationError(404, "action_not_found", "Automation action was not found.")
            if row["state"] == "unknown":
                matches = await self.provider.find_tasks(self.config.project_id, row["marker"])
                factual: list[str] = []
                for match in matches:
                    try:
                        factual.append(self._factual_id(match.get("id"), "externalId"))
                    except AutomationError:
                        # Malformed provider rows cannot establish success.
                        # Ignore them and leave the action unknown.
                        continue
                if len(factual) == 1:
                    try:
                        self._conn.execute("BEGIN IMMEDIATE")
                        self._conn.execute("UPDATE actions SET state='succeeded',external_id=?,updated_at=? WHERE action_id=?",
                                           (factual[0], time.time(), row["action_id"]))
                        self._audit("execute_reconciled", "succeeded", preview_id=preview_id,
                                    action_id=row["action_id"], external_id=factual[0])
                        self._conn.execute("COMMIT")
                    except Exception:
                        if self._conn.in_transaction:
                            self._conn.execute("ROLLBACK")
                        raise AutomationError(503, "audit_unavailable", "Mandatory automation audit append failed.")
                    row = self._conn.execute("SELECT * FROM actions WHERE action_id=?", (row["action_id"],)).fetchone()
            # Ambiguous compensating DELETE is reconciled by proving absence.
            if row["reversal_state"] in {"reversing", "reversal_unknown"} and row["external_id"]:
                if not await self.provider.task_exists(row["external_id"], self.config.project_id):
                    try:
                        self._conn.execute("BEGIN IMMEDIATE")
                        self._conn.execute("UPDATE actions SET reversal_state='reversed',updated_at=? WHERE action_id=?",
                                           (time.time(), row["action_id"]))
                        self._audit("reversal_reconciled", "reversed", preview_id=preview_id,
                                    action_id=row["action_id"], external_id=row["external_id"],
                                    reversal_action_id=row["reversal_action_id"])
                        self._conn.execute("COMMIT")
                    except Exception:
                        if self._conn.in_transaction:
                            self._conn.execute("ROLLBACK")
                        raise AutomationError(503, "audit_unavailable", "Mandatory automation audit append failed.")
                    row = self._conn.execute("SELECT * FROM actions WHERE action_id=?", (row["action_id"],)).fetchone()
            response = self._action_response(row, "status")
            return (202 if response["state"] in {"unknown", "reversal_unknown", "executing", "reversing"} else 200), response

    async def reverse(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if set(body) != {"version", "operation", "actionId", "reversalActionId"}:
            raise AutomationError(400, "invalid_reverse_shape", "Reverse accepts only version, operation, actionId, and reversalActionId.")
        action_id = self._factual_id(body.get("actionId"), "actionId")
        reversal_id = self._factual_id(body.get("reversalActionId"), "reversalActionId")
        async with self._lock:
            row = self._conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row or row["state"] != "succeeded" or not row["external_id"]:
                raise AutomationError(409, "action_not_reversible", "Only a factually succeeded action can be reversed.")
            if row["reversal_action_id"] != reversal_id:
                raise AutomationError(409, "reversal_conflict", "Action is bound to a different factual reversal identifier.")
            if row["reversal_state"]:
                response = self._action_response(row, "reverse")
                return (202 if response["state"] in {"reversing", "reversal_unknown"} else 200), response
            request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, action_id + ":" + reversal_id))
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("UPDATE actions SET reversal_state='reversing',reversal_request_id=?,updated_at=? WHERE action_id=?",
                                   (request_id, time.time(), action_id))
                self._audit("reversal_authorized", "reversing", preview_id=row["preview_id"],
                            action_id=action_id, external_id=row["external_id"], reversal_action_id=reversal_id)
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise AutomationError(503, "audit_unavailable", "Mandatory automation audit append failed.")
            try:
                await self.provider.delete_task(row["external_id"], request_id)
                exists = await self.provider.task_exists(row["external_id"], self.config.project_id)
                state, status = ("reversal_failed", 502) if exists else ("reversed", 200)
            except Exception:
                # DELETE has begun. Any transport, response, or verification
                # failure after this point is uncertain; do not issue a second
                # DELETE. Status reconciliation proves provider absence.
                state, status = "reversal_unknown", 202
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("UPDATE actions SET reversal_state=?,updated_at=? WHERE action_id=?", (state, time.time(), action_id))
                self._audit("reversal_result", state, preview_id=row["preview_id"], action_id=action_id,
                            external_id=row["external_id"], reversal_action_id=reversal_id)
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                self._conn.execute("UPDATE actions SET reversal_state='reversal_unknown',updated_at=? WHERE action_id=?", (time.time(), action_id))
                status = 202
            row = self._conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return status, self._action_response(row, "reverse")

    def audit_rows(self) -> list[dict[str, Any]]:
        """Test/diagnostic helper; schema itself guarantees allowlisted fields."""
        assert self._conn
        return [dict(row) for row in self._conn.execute("SELECT * FROM audit ORDER BY sequence")]
