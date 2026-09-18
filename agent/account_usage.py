from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import httpx

from agent.anthropic_adapter import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class AccountUsageSnapshot:
    provider: str
    source: str
    fetched_at: datetime
    title: str = "Account limits"
    plan: Optional[str] = None
    windows: tuple[AccountUsageWindow, ...] = ()
    details: tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.windows or self.details) and not self.unavailable_reason


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.replace("_", " ").replace("-", " ").title()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    local_dt = dt.astimezone()
    delta = dt - _utc_now()
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return f"now ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        rel = f"in {days}d {hours}h"
    elif hours > 0:
        rel = f"in {hours}h {minutes}m"
    else:
        rel = f"in {minutes}m"
    return f"{rel} ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    header = f"📈 {'**' if markdown else ''}{snapshot.title}{'**' if markdown else ''}"
    lines = [header]
    if snapshot.plan:
        lines.append(f"Provider: {snapshot.provider} ({snapshot.plan})")
    else:
        lines.append(f"Provider: {snapshot.provider}")
    for window in snapshot.windows:
        if window.used_percent is None:
            base = f"{window.label}: unavailable"
        else:
            remaining = max(0, round(100 - float(window.used_percent)))
            used = max(0, round(float(window.used_percent)))
            base = f"{window.label}: {remaining}% remaining ({used}% used)"
        if window.reset_at:
            base += f" • resets {_format_reset(window.reset_at)}"
        elif window.detail:
            base += f" • {window.detail}"
        lines.append(base)
    for detail in snapshot.details:
        lines.append(detail)
    if snapshot.unavailable_reason:
        lines.append(f"Unavailable: {snapshot.unavailable_reason}")
    return lines


def _fmt_usd(d: float) -> str:
    return f"${d:,.2f}"


def _is_finite_num(v: Any) -> TypeGuard[float]:
    """True iff v is a real numeric value (int or float, not bool, not NaN/Inf).

    Typed as a ``TypeGuard[float]`` so the type checker narrows ``v`` to a real
    number in the positive branch — callers can then do arithmetic / pass it to
    ``_fmt_usd`` without a None-operand warning.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def build_nous_credits_snapshot(account_info) -> Optional[AccountUsageSnapshot]:
    """Map a NousPortalAccountInfo into an AccountUsageSnapshot for /usage.

    Shows dollar magnitudes (subscription / top-up / total) + renewal date + a
    portal CTA. When the portal supplies a subscription denominator
    (``monthly_credits``), also emits a subscription-usage window so the renderer
    shows a real ``% used`` gauge; when it's absent (older portals) the view
    gracefully degrades to magnitudes-only. Returns None when there's no usable
    account info to show (fail-open: caller just shows nothing).
    """
    try:
        from hermes_cli.nous_account import nous_portal_topup_url

        if account_info is None or not getattr(account_info, "logged_in", False):
            return None

        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)

        windows: list[AccountUsageWindow] = []
        details: list[str] = []

        # Subscription usage gauge — only when the portal supplies a positive
        # monthly_credits denominator AND a finite remaining balance that does
        # not exceed the cap. Money math is on float dollars (allowed: numeric
        # account fields, NOT a server-provided *_usd string). used = cap -
        # remaining; clamp [0,100] so a debt balance (remaining < 0) reads 100%.
        # Excluded on purpose:
        #   - non-finite values (NaN/Infinity slip past isinstance and json.loads
        #     parses bare NaN/Infinity by default) → would render "$nan"/"$inf"
        #     and a falsely-confident gauge;
        #   - remaining > cap (rollover balance spanning the period) → monthly_credits
        #     is no longer a meaningful denominator, and "$X of $Y left" with X>Y
        #     reads as a contradiction. Both fall back to the magnitudes lines.
        if sub is not None:
            monthly_credits = getattr(sub, "monthly_credits", None)
            sub_remaining = getattr(sub, "credits_remaining", None)
            if (
                _is_finite_num(monthly_credits)
                and monthly_credits > 0
                and _is_finite_num(sub_remaining)
                and sub_remaining <= monthly_credits
            ):
                used = monthly_credits - sub_remaining
                used_pct = max(0.0, min(100.0, used / monthly_credits * 100.0))
                windows.append(
                    AccountUsageWindow(
                        label="Subscription",
                        used_percent=used_pct,
                        detail=f"{_fmt_usd(sub_remaining)} of {_fmt_usd(monthly_credits)} left",
                    )
                )

        if access is not None:
            sub_credits = getattr(access, "subscription_credits_remaining", None)
            if _is_finite_num(sub_credits):
                details.append(f"Subscription credits: {_fmt_usd(sub_credits)}")
            purchased = getattr(access, "purchased_credits_remaining", None)
            if _is_finite_num(purchased):
                details.append(f"Top-up credits: {_fmt_usd(purchased)}")
            total_usable = getattr(access, "total_usable_credits", None)
            if _is_finite_num(total_usable):
                details.append(f"Total usable: {_fmt_usd(total_usable)}")

        if sub is not None:
            rollover = getattr(sub, "rollover_credits", None)
            if _is_finite_num(rollover) and rollover > 0:
                details.append(f"Rollover: {_fmt_usd(rollover)}")
            period_end = getattr(sub, "current_period_end", None)
            if period_end:
                details.append(f"Renews: {period_end}")

        paid = getattr(account_info, "paid_service_access", None)
        if paid is False:
            details.append("Status: access depleted — top up to restore")

        if not windows and not details:
            return None

        details.append(f"Top up: {nous_portal_topup_url(account_info)}")
        details.append("(or run /topup)")

        plan = getattr(sub, "plan", None) if sub is not None else None
        return AccountUsageSnapshot(
            provider="nous",
            source="portal-account",
            fetched_at=_utc_now(),
            title="Nous credits",
            plan=plan,
            windows=tuple(windows),
            details=tuple(details),
        )
    except (AttributeError, TypeError):
        return None


def nous_credits_lines(*, markdown: bool = False, timeout: float = 10.0) -> list[str]:
    """Return rendered Nous-credits /usage lines, or [] when there's nothing to show.

    Account-independent of any live agent: gated on "a Nous account is logged in"
    (a cheap local auth-state check), then a wall-clock-bounded portal fetch. Shared
    by the CLI ``_show_usage`` and the TUI ``session.usage`` RPC so both surfaces show
    the same block regardless of session API-call count or resume state. Fail-open:
    any auth/portal hiccup or timeout returns [] (the caller shows nothing).

    Dev override: when HERMES_DEV_CREDITS_FIXTURE selects a fixture state, /usage
    renders from that fixture instead of the real portal (so the block + gauge are
    testable without a live account). Throwaway scaffolding.
    """
    # Dev fixture short-circuit — render /usage from the injected state, no portal.
    try:
        from agent.credits_tracker import dev_fixture_credits_state

        fixture = dev_fixture_credits_state()
    except Exception:
        fixture = None
    if fixture is not None:
        snapshot = _snapshot_from_credits_state(fixture)
        return render_account_usage_lines(snapshot, markdown=markdown)

    try:
        from hermes_cli.auth import get_provider_auth_state

        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        if not (isinstance(tok, str) and tok.strip()):
            return []
    except Exception:
        return []
    try:
        import concurrent.futures

        from hermes_cli.nous_account import get_nous_portal_account_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            account = pool.submit(
                get_nous_portal_account_info, force_fresh=True
            ).result(timeout=timeout)
        snapshot = build_nous_credits_snapshot(account)
        return render_account_usage_lines(snapshot, markdown=markdown)
    except Exception:
        # Fail-open (caller shows nothing), but leave a breadcrumb so a dead
        # /usage credits block is diagnosable in agent.log without a dev flag.
        logger.debug("credits ▸ /usage portal fetch/render failed (fail-open)", exc_info=True)
        return []


def _snapshot_from_credits_state(state) -> Optional[AccountUsageSnapshot]:
    """Map a header-shaped CreditsState (e.g. a dev fixture) to the /usage snapshot.

    Renders the same magnitudes + monthly-grant % window the portal path produces,
    so HERMES_DEV_CREDITS_FIXTURE can exercise /usage without a live account. The
    *_usd strings are mock display values here (not server balance to compute on);
    the % comes from CreditsState.used_fraction (micros math). Fail-open → None.
    """
    try:
        if state is None:
            return None

        windows: list[AccountUsageWindow] = []
        details: list[str] = []

        uf = getattr(state, "used_fraction", None)
        if isinstance(uf, (int, float)) and math.isfinite(uf):
            cap_usd = getattr(state, "subscription_limit_usd", None)
            sub_usd = getattr(state, "subscription_usd", None)
            detail = None
            if sub_usd and cap_usd:
                detail = f"${sub_usd} of ${cap_usd} left"
            windows.append(
                AccountUsageWindow(
                    label="Subscription",
                    used_percent=max(0.0, min(100.0, uf * 100.0)),
                    detail=detail,
                )
            )

        sub_usd = getattr(state, "subscription_usd", None)
        if sub_usd:
            details.append(f"Subscription credits: ${sub_usd}")
        purchased_usd = getattr(state, "purchased_usd", None)
        if purchased_usd:
            details.append(f"Top-up credits: ${purchased_usd}")
        remaining_usd = getattr(state, "remaining_usd", None)
        if remaining_usd:
            details.append(f"Total usable: ${remaining_usd}")
        if getattr(state, "paid_access", True) is False:
            details.append("Status: access depleted — top up to restore")

        if not windows and not details:
            return None

        details.append("(dev fixture — HERMES_DEV_CREDITS_FIXTURE)")
        return AccountUsageSnapshot(
            provider="nous",
            source="dev-fixture",
            fetched_at=_utc_now(),
            title="Nous credits",
            windows=tuple(windows),
            details=tuple(details),
        )
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True)
class CreditsView:
    """Surface-agnostic data for the ``/topup`` balance view.

    One portal fetch, one parse — consumed identically by the CLI panel, the
    gateway button, and any other money surface. Fail-open: when not logged in
    or the portal is unreachable, ``logged_in`` is False / ``topup_url`` is None
    and callers degrade gracefully.
    """

    logged_in: bool
    balance_lines: tuple[str, ...] = ()
    identity_line: Optional[str] = None
    topup_url: Optional[str] = None
    depleted: bool = False


def build_credits_view(*, markdown: bool = False, timeout: float = 10.0) -> CreditsView:
    """Build the /topup balance view: balance block + identity line + top-up URL.

    Reuses the same account fetch + snapshot + URL builder as the /usage credits
    block, so the numbers always match. The balance block is the rendered
    snapshot MINUS its trailing top-up/command-hint lines (the /topup surface
    supplies its own affordance). Fail-open → ``CreditsView(logged_in=False)``.
    """
    not_logged_in = CreditsView(logged_in=False)
    try:
        from hermes_cli.auth import get_provider_auth_state

        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        if not (isinstance(tok, str) and tok.strip()):
            return not_logged_in
    except Exception:
        return not_logged_in

    try:
        import concurrent.futures

        from hermes_cli.nous_account import (
            get_nous_portal_account_info,
            nous_portal_topup_url,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            account = pool.submit(get_nous_portal_account_info, force_fresh=True).result(
                timeout=timeout
            )
    except Exception:
        logger.debug("credits ▸ /topup portal fetch failed (fail-open)", exc_info=True)
        return not_logged_in

    if account is None or not getattr(account, "logged_in", False):
        return not_logged_in

    snapshot = build_nous_credits_snapshot(account)
    # Balance lines = the snapshot block minus the two trailing affordance lines
    # ("Top up: <url>" + "(or run /topup)") that build_nous_credits_snapshot
    # appends for the /usage surface. /topup renders its own button/panel.
    balance_lines: list[str] = []
    if snapshot is not None:
        rendered = render_account_usage_lines(snapshot, markdown=markdown)
        balance_lines = [
            line
            for line in rendered
            if not line.lstrip().startswith("Top up:")
            and not line.lstrip().startswith("(or run")
        ]

    # Identity line — shown before any open (roadmap §4.4).
    email = getattr(account, "email", None)
    org_name = getattr(account, "org_name", None)
    who: list[str] = []
    if email:
        who.append(str(email))
    if org_name:
        who.append(f"org {org_name}")
    identity_line = ("Topping up as " + " / ".join(who)) if who else None

    return CreditsView(
        logged_in=True,
        balance_lines=tuple(balance_lines),
        identity_line=identity_line,
        topup_url=nous_portal_topup_url(account),
        depleted=getattr(account, "paid_service_access", None) is False,
    )


def _codex_backend_urls(base_url: str) -> tuple[str, str, str]:
    """Resolve the Codex backend endpoints (usage, reset-credits list, consume).

    Mirrors the Codex CLI's PathStyle split (codex-rs backend-client): base URLs
    containing ``/backend-api`` use the ChatGPT ``/wham/...`` paths; everything
    else uses ``/api/codex/...``.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return (
        prefix + "/usage",
        prefix + "/rate-limit-reset-credits",
        prefix + "/rate-limit-reset-credits/consume",
    )


def _resolve_codex_usage_url(base_url: str) -> str:
    return _codex_backend_urls(base_url)[0]


def _resolve_codex_usage_credentials(
    base_url: Optional[str],
    api_key: Optional[str],
) -> tuple[str, str, Optional[str]]:
    """Resolve Codex quota credentials from the native runtime path.

    Prefer explicit live-agent credentials, then the legacy singleton OAuth
    state, then the credential pool.  Hermes's native OAuth setup now stores
    device-code logins in the pool, so quota diagnostics must not depend only
    on the older singleton store.
    """
    explicit_key = str(api_key or "").strip()
    if explicit_key:
        return explicit_key, str(base_url or "").strip(), None

    # Tier 2: the native runtime resolver. It ALREADY falls back to the
    # credential pool when the singleton is empty (see
    # ``resolve_codex_runtime_credentials`` — issue #32992), so in a pool-only
    # setup this returns a usable ``source="credential_pool"`` token.
    #
    # Only ``AuthError`` ("no creds" / rate-limited) is caught so tier 3 can
    # run: a broad ``except Exception`` would (a) mask a transient refresh /
    # network failure and silently hand back a DIFFERENT pool account's usage,
    # and (b) hide genuine programming errors. A refresh/network error must
    # propagate — the outer ``fetch_account_usage`` guard fails open (shows
    # nothing this turn) rather than reporting the wrong account.
    #
    # The ``account_id`` (for the ``ChatGPT-Account-Id`` header) is read
    # best-effort: a partial/missing singleton token store must not sink an
    # otherwise-usable resolver credential and force a header-less pool fallback.
    try:
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        account_id: Optional[str] = None
        try:
            token_data = _read_codex_tokens()
            tokens = token_data.get("tokens") or {}
            account_id = str(tokens.get("account_id", "") or "").strip() or None
        except AuthError:
            # Pool-only creds carry no singleton account_id; header is optional.
            logger.debug("codex ▸ /usage account_id read failed (best-effort)", exc_info=True)
        return creds["api_key"], str(creds.get("base_url", "") or "").strip(), account_id
    except AuthError:
        logger.debug("codex ▸ /usage runtime resolver returned no creds; trying pool", exc_info=True)

    # Tier 3: direct pool select. Reached only when the resolver itself raises
    # AuthError (e.g. singleton missing AND its own pool read found nothing at
    # resolve time, but a pool entry is usable now). Pool credentials have no
    # account_id concept, so the ChatGPT-Account-Id header is intentionally
    # omitted here.
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    if entry is None:
        raise RuntimeError("No available openai-codex credential in credential pool")
    return entry.runtime_api_key, str(entry.runtime_base_url or base_url or "").strip(), None


def _fetch_codex_account_usage(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_resolve_codex_usage_url(resolved_base_url), headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    rate_limit = payload.get("rate_limit") or {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("primary_window", "Session"), ("secondary_window", "Weekly")):
        window = rate_limit.get(key) or {}
        used = window.get("used_percent")
        if used is None:
            continue
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=float(used),
                reset_at=_parse_dt(window.get("reset_at")),
            )
        )
    details: list[str] = []
    reset_credits = payload.get("rate_limit_reset_credits") or {}
    banked = reset_credits.get("available_count")
    if isinstance(banked, (int, float)) and int(banked) > 0:
        count = int(banked)
        plural = "s" if count != 1 else ""
        details.append(
            f"You have {count} reset{plural} banked - use /usage reset to activate"
        )
    credits = payload.get("credits") or {}
    if credits.get("has_credits"):
        balance = credits.get("balance")
        if isinstance(balance, (int, float)):
            details.append(f"Credits balance: ${float(balance):.2f}")
        elif credits.get("unlimited"):
            details.append("Credits balance: unlimited")
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=_utc_now(),
        plan=_title_case_slug(payload.get("plan_type")),
        windows=tuple(windows),
        details=tuple(details),
    )


@dataclass(frozen=True)
class CodexResetRedeemResult:
    """Outcome of a `/usage reset` attempt against the Codex backend."""

    status: str  # reset | nothing_to_reset | no_credit | already_redeemed |
    #              not_exhausted | no_credits_banked | unavailable
    message: str
    available_count: int = 0
    windows_reset: int = 0

    @property
    def redeemed(self) -> bool:
        return self.status == "reset"


# Client-side guard threshold: a rate-limit window only counts as exhausted
# when it is fully used. Below this, redeeming a banked reset wastes most of
# its value, so we block and point at --force instead.
_CODEX_WINDOW_EXHAUSTED_PERCENT = 100.0


def redeem_codex_reset_credit(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    force: bool = False,
) -> CodexResetRedeemResult:
    """Redeem one banked Codex rate-limit reset credit (`/usage reset`).

    Flow (mirrors the Codex CLI's reset-credits picker, codex-rs
    ``backend-client``):

    1. ``GET .../usage`` — read the current windows + banked credit count.
    2. Guard: zero banked credits → refuse. No window fully used and not
       ``force`` → refuse with a warning (a banked reset restores the WHOLE
       5h + weekly allowance; burning it early wastes it). The backend has
       the same protection (``nothing_to_reset`` doesn't consume the
       credit), but failing fast client-side gives a clearer message.
    3. ``POST .../rate-limit-reset-credits/consume`` with a fresh UUID
       idempotency key (``redeem_request_id``). No ``credit_id`` — the
       backend picks the next available credit, exactly like the CLI's
       default "Full reset" option.

    Never raises: every failure mode returns a ``CodexResetRedeemResult``
    with a user-renderable message.
    """
    import uuid

    try:
        token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    except Exception:
        return CodexResetRedeemResult(
            status="unavailable",
            message="No Codex credentials available. Run `hermes auth` to sign in with your ChatGPT account.",
        )
    usage_url, _credits_url, consume_url = _codex_backend_urls(resolved_base_url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    try:
        with httpx.Client(timeout=15.0) as client:
            usage_resp = client.get(usage_url, headers=headers)
            usage_resp.raise_for_status()
            payload = usage_resp.json() or {}

            reset_credits = payload.get("rate_limit_reset_credits") or {}
            raw_count = reset_credits.get("available_count")
            available = int(raw_count) if isinstance(raw_count, (int, float)) else 0
            if available <= 0:
                return CodexResetRedeemResult(
                    status="no_credits_banked",
                    message="No banked reset credits on this account — nothing to redeem.",
                )

            rate_limit = payload.get("rate_limit") or {}
            worst_used: Optional[float] = None
            for key in ("primary_window", "secondary_window"):
                used = (rate_limit.get(key) or {}).get("used_percent")
                if isinstance(used, (int, float)):
                    worst_used = max(worst_used or 0.0, float(used))
            exhausted = worst_used is not None and worst_used >= _CODEX_WINDOW_EXHAUSTED_PERCENT
            if not exhausted and not force:
                usage_note = (
                    f"your busiest window is only {worst_used:.0f}% used"
                    if worst_used is not None
                    else "your current usage could not be confirmed as exhausted"
                )
                plural = "s" if available != 1 else ""
                return CodexResetRedeemResult(
                    status="not_exhausted",
                    message=(
                        f"⚠️ Not redeeming: {usage_note}. A banked reset restores your FULL "
                        f"5h + weekly limits, so spending it now would waste most of it. "
                        f"You have {available} reset{plural} banked. "
                        f"Use `/usage reset --force` to redeem anyway."
                    ),
                    available_count=available,
                )

            consume_resp = client.post(
                consume_url,
                headers={**headers, "Content-Type": "application/json"},
                json={"redeem_request_id": str(uuid.uuid4())},
            )
            consume_resp.raise_for_status()
            body = consume_resp.json() or {}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            return CodexResetRedeemResult(
                status="unavailable",
                message=(
                    "Codex backend rejected the request (HTTP "
                    f"{code}). Reset credits require ChatGPT-account (OAuth) auth — "
                    "run `hermes auth` and sign in with your ChatGPT account."
                ),
            )
        return CodexResetRedeemResult(
            status="unavailable",
            message=f"Codex backend error (HTTP {code}) — try again shortly.",
        )
    except Exception as exc:
        return CodexResetRedeemResult(
            status="unavailable",
            message=f"Could not reach the Codex backend: {exc}",
        )

    code = str(body.get("code", "") or "").strip().lower()
    windows_reset = body.get("windows_reset")
    windows_reset = int(windows_reset) if isinstance(windows_reset, (int, float)) else 0
    remaining = max(0, available - 1)
    plural = "s" if remaining != 1 else ""
    if code == "reset":
        # The redeemed reset restores the account's quota upstream — lift any
        # persisted pool cooldowns so Hermes doesn't keep the credential
        # frozen behind the now-stale ``last_error_reset_at`` (issue #43747).
        try:
            from hermes_cli.auth import clear_codex_pool_quota_cooldowns

            clear_codex_pool_quota_cooldowns()
        except Exception:
            logger.debug(
                "Failed to clear Codex pool cooldowns after reset redemption",
                exc_info=True,
            )
        return CodexResetRedeemResult(
            status="reset",
            message=(
                f"✅ Reset redeemed — your usage limits have been reset. "
                f"{remaining} banked reset{plural} remaining."
            ),
            available_count=remaining,
            windows_reset=windows_reset,
        )
    if code == "nothing_to_reset":
        return CodexResetRedeemResult(
            status="nothing_to_reset",
            message=(
                "Backend reports nothing to reset — your limits aren't exhausted. "
                "The credit was NOT spent."
            ),
            available_count=available,
        )
    if code == "no_credit":
        return CodexResetRedeemResult(
            status="no_credit",
            message="Backend reports no available reset credit on this account.",
        )
    if code == "already_redeemed":
        return CodexResetRedeemResult(
            status="already_redeemed",
            message="This redemption was already processed — no additional credit was spent.",
            available_count=remaining,
        )
    return CodexResetRedeemResult(
        status="unavailable",
        message=f"Unexpected response from the Codex backend: {body!r}",
    )


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    token = (resolve_anthropic_token() or "").strip()
    if not token:
        return None
    if not _is_oauth_token(token):
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.",
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": "claude-code/2.1.0",
    }
    with httpx.Client(timeout=15.0) as client:
        response = client.get("https://api.anthropic.com/api/oauth/usage", headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    windows: list[AccountUsageWindow] = []
    mapping = (
        ("five_hour", "Current session"),
        ("seven_day", "Current week"),
        ("seven_day_opus", "Opus week"),
        ("seven_day_sonnet", "Sonnet week"),
    )
    for key, label in mapping:
        window = payload.get(key) or {}
        util = window.get("utilization")
        if util is None:
            continue
        used = float(util) * 100 if float(util) <= 1 else float(util)
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=used,
                reset_at=_parse_dt(window.get("resets_at")),
            )
        )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    if extra.get("is_enabled"):
        used_credits = extra.get("used_credits")
        monthly_limit = extra.get("monthly_limit")
        currency = extra.get("currency") or "USD"
        if isinstance(used_credits, (int, float)) and isinstance(monthly_limit, (int, float)):
            details.append(
                f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {currency}"
            )
    return AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(
        requested="openrouter",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    normalized = str(runtime.get("base_url", "") or "").rstrip("/")
    credits_url = f"{normalized}/credits"
    key_url = f"{normalized}/key"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        credits_resp = client.get(credits_url, headers=headers)
        credits_resp.raise_for_status()
        credits = (credits_resp.json() or {}).get("data") or {}
        try:
            key_resp = client.get(key_url, headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
        except Exception:
            key_data = {}
    total_credits = float(credits.get("total_credits") or 0.0)
    total_usage = float(credits.get("total_usage") or 0.0)
    details = [f"Credits balance: ${max(0.0, total_credits - total_usage):.2f}"]
    windows: list[AccountUsageWindow] = []
    limit = key_data.get("limit")
    limit_remaining = key_data.get("limit_remaining")
    limit_reset = str(key_data.get("limit_reset") or "").strip()
    usage = key_data.get("usage")
    if (
        isinstance(limit, (int, float))
        and float(limit) > 0
        and isinstance(limit_remaining, (int, float))
        and 0 <= float(limit_remaining) <= float(limit)
    ):
        limit_value = float(limit)
        remaining_value = float(limit_remaining)
        used_percent = ((limit_value - remaining_value) / limit_value) * 100
        detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining"]
        if limit_reset:
            detail_parts.append(f"resets {limit_reset}")
        windows.append(
            AccountUsageWindow(
                label="API key quota",
                used_percent=used_percent,
                detail=" • ".join(detail_parts),
            )
        )
    if isinstance(usage, (int, float)):
        usage_parts = [f"API key usage: ${float(usage):.2f} total"]
        for value, label in (
            (key_data.get("usage_daily"), "today"),
            (key_data.get("usage_weekly"), "this week"),
            (key_data.get("usage_monthly"), "this month"),
        ):
            if isinstance(value, (int, float)) and float(value) > 0:
                usage_parts.append(f"${float(value):.2f} {label}")
        details.append(" • ".join(usage_parts))
    return AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


# ---------------------------------------------------------------------------
# Claude Code (the Claude subscription behind the local CLI)
# ---------------------------------------------------------------------------

# Plan windows as Claude Code's ``get_usage`` names them (utilization in
# percent) and as ``rate_limit_info.unifiedWindows`` does (as a fraction).
_CLAUDE_WINDOW_LABELS: tuple[tuple[str, str], ...] = (
    ("five_hour", "Current session (5h)"),
    ("seven_day", "Current week"),
    ("seven_day_opus", "Opus week"),
    ("seven_day_sonnet", "Sonnet week"),
    ("seven_day_overage_included", "Week incl. extra usage"),
)
# ``get_usage`` asks claude.ai; /usage and /claude usage reuse an answer this
# long instead of starting the CLI again.
_CLAUDE_USAGE_CACHE_SECONDS = 60.0
_claude_usage_cache: Optional[tuple[float, AccountUsageSnapshot]] = None


def _claude_money(amount: Any, decimals: Any) -> Optional[float]:
    if not _is_finite_num(amount):
        return None
    places = decimals if isinstance(decimals, int) and not isinstance(decimals, bool) else 2
    return float(amount) / (10 ** max(0, min(places, 6)))


def _claude_extra_usage_detail(extra: Any) -> Optional[str]:
    if not isinstance(extra, dict):
        return None
    currency = str(extra.get("currency") or "USD")
    used = _claude_money(extra.get("used_credits"), extra.get("decimal_places"))
    limit = _claude_money(extra.get("monthly_limit"), extra.get("decimal_places"))
    amounts = f"{used:.2f} / {limit:.2f} {currency}" if used is not None and limit is not None else ""
    if extra.get("is_enabled"):
        return f"Extra usage: {amounts}" if amounts else "Extra usage: on"
    reason = str(extra.get("disabled_reason") or "").replace("_", " ").strip()
    if not reason and not amounts:
        return None
    state = f"off ({reason})" if reason else "off"
    return f"Extra usage: {state}" + (f" · {amounts} used" if amounts else "")


def _claude_model_scoped_windows(limits: dict[str, Any]) -> list[tuple[str, float, Any]]:
    """``(model name, used percent, resets_at)`` of per-model weekly limits.

    ``model_scoped`` lists them when present; otherwise the ``limits`` rows
    of kind ``weekly_scoped`` name the model in ``scope.model``.
    """

    found: list[tuple[str, float, Any]] = []
    for scoped in limits.get("model_scoped") or []:
        if isinstance(scoped, dict) and _is_finite_num(scoped.get("utilization")):
            name = str(scoped.get("display_name") or "Model").strip()
            found.append((name, float(scoped["utilization"]), scoped.get("resets_at")))
    if found:
        return found
    for row in limits.get("limits") or []:
        if not isinstance(row, dict) or row.get("kind") != "weekly_scoped":
            continue
        if not _is_finite_num(row.get("percent")):
            continue
        scope = row.get("scope") if isinstance(row.get("scope"), dict) else {}
        model = scope.get("model") if isinstance(scope.get("model"), dict) else {}
        name = str(model.get("display_name") or model.get("id") or "Model").strip()
        found.append((name, float(row["percent"]), row.get("resets_at")))
    return found


def _claude_plan_label(subscription_type: Any) -> Optional[str]:
    return _title_case_slug(subscription_type) if isinstance(subscription_type, str) else None


def build_claude_code_usage_snapshot(
    usage: Optional[dict[str, Any]] = None,
    *,
    rate_limit: Optional[dict[str, Any]] = None,
    recorded_at: Optional[float] = None,
    note: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    """Claude plan windows for /usage.

    ``usage`` is Claude Code's ``get_usage`` answer (live: session and weekly
    windows, per-model weekly limits, extra usage, plan). Without it,
    ``rate_limit`` is the CLI's last ``rate_limit_info`` (session and weekly
    windows as of ``recorded_at``). ``note`` explains why live data is
    missing. None when neither says anything.
    """

    windows: list[AccountUsageWindow] = []
    details: list[str] = []
    if isinstance(usage, dict):
        limits = usage.get("rate_limits") if isinstance(usage.get("rate_limits"), dict) else {}
        for key, label in _CLAUDE_WINDOW_LABELS:
            window = limits.get(key)
            if isinstance(window, dict) and _is_finite_num(window.get("utilization")):
                windows.append(
                    AccountUsageWindow(
                        label=label,
                        used_percent=float(window["utilization"]),
                        reset_at=_parse_dt(window.get("resets_at")),
                    )
                )
        for name, used, resets_at in _claude_model_scoped_windows(limits):
            windows.append(
                AccountUsageWindow(label=f"{name} week", used_percent=used, reset_at=_parse_dt(resets_at))
            )
        extra = _claude_extra_usage_detail(limits.get("extra_usage"))
        if extra:
            details.append(extra)
        unavailable = None
        if usage.get("rate_limits_available") is False:
            unavailable = "Claude Code reports no plan limits for this account"
        if not windows and not details and not unavailable:
            return None
        return AccountUsageSnapshot(
            provider="claude-code",
            source="claude_code_get_usage",
            fetched_at=_utc_now(),
            title="Claude plan limits",
            plan=_claude_plan_label(usage.get("subscription_type")),
            windows=tuple(windows),
            details=tuple(details),
            unavailable_reason=unavailable,
        )

    if not isinstance(rate_limit, dict) or not rate_limit:
        if not note:
            return None
        return AccountUsageSnapshot(
            provider="claude-code",
            source="claude_code_rate_limit",
            fetched_at=_utc_now(),
            title="Claude plan limits",
            unavailable_reason=note,
        )
    from agent.claude_code_session import _RATE_LIMIT_LABELS, plan_status_live, plan_windows

    labels = dict(_CLAUDE_WINDOW_LABELS)
    now = time.time()
    for key, (utilization, resets_at) in plan_windows(rate_limit).items():
        if utilization is None:
            continue
        label = labels.get(key) or key.replace("_", " ").capitalize()
        if resets_at is not None and resets_at <= now:
            # The report predates this window's reset: its figure is over.
            stamp = datetime.fromtimestamp(resets_at, tz=timezone.utc).astimezone()
            details.append(f"{label}: reset at {stamp.strftime('%Y-%m-%d %H:%M %Z')}, after this report")
            continue
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=utilization * 100.0,
                reset_at=_parse_dt(resets_at),
            )
        )
    status = str(rate_limit.get("status") or "") if plan_status_live(rate_limit, now) else ""
    if status and status != "allowed":
        kind = str(rate_limit.get("rateLimitType") or "")
        label = _RATE_LIMIT_LABELS.get(kind) or kind.replace("_", " ")
        details.append(f"Status: {status.replace('_', ' ')}" + (f" ({label})" if label else ""))
    overage = str(rate_limit.get("overageStatus") or "")
    if overage:
        reason = str(rate_limit.get("overageDisabledReason") or "").replace("_", " ")
        details.append(f"Extra usage: {overage.replace('_', ' ')}" + (f" ({reason})" if reason else ""))
    if recorded_at:
        stamp = datetime.fromtimestamp(float(recorded_at), tz=timezone.utc).astimezone()
        details.append(f"As reported by Claude Code at {stamp.strftime('%Y-%m-%d %H:%M %Z')}")
    if note:
        details.append(f"Live numbers unavailable: {note}")
    if not windows and not details:
        return None
    return AccountUsageSnapshot(
        provider="claude-code",
        source="claude_code_rate_limit",
        fetched_at=_utc_now(),
        title="Claude plan limits",
        windows=tuple(windows),
        details=tuple(details),
    )


def fetch_claude_code_account_usage(
    client: Any = None,
    *,
    timeout: float = 30.0,
    use_cache: bool = True,
) -> Optional[AccountUsageSnapshot]:
    """Claude plan windows from Claude Code itself (no credentials read).

    Asks the CLI (``get_usage`` control request; ``client``'s warm process
    when it is idle, else a throwaway control-only process, never a model
    turn). When that fails, falls back to the CLI's last reported
    ``rate_limit_info``: ``client``'s session, else the persisted copy.
    """

    global _claude_usage_cache
    now = time.monotonic()
    cached = _claude_usage_cache
    if use_cache and cached is not None and now - cached[0] < _CLAUDE_USAGE_CACHE_SECONDS:
        return cached[1]
    from agent import claude_code_session as ccs

    request = [("get_usage", {"skip_behaviors": True})]
    note: Optional[str] = None
    try:
        if client is not None and callable(getattr(client, "control_requests", None)):
            answer = client.control_requests(request, timeout=timeout)[0]
        else:
            from agent.claude_code_client import _build_subprocess_env, _resolve_command

            answer = ccs.run_control_requests(
                request,
                command=_resolve_command(),
                env=_build_subprocess_env(),
                timeout=timeout,
            )[0]
    except Exception as exc:  # launch failures, OS errors
        answer = exc
    if isinstance(answer, dict):
        snapshot = build_claude_code_usage_snapshot(answer)
        if snapshot is not None:
            _claude_usage_cache = (now, snapshot)
            return snapshot
        note = "Claude Code returned no plan data"
    else:
        note = str(answer)[:200]
        logger.debug("claude-code ▸ get_usage failed: %s", note)

    # Every session persists what it records, so the file is at least as
    # recent as any session's copy (another profile may have written later).
    info: dict[str, Any] = {}
    recorded_at: Optional[float] = None
    persisted = ccs.load_rate_limit_snapshot()
    if persisted is not None:
        info, recorded_at = dict(persisted["info"]), persisted.get("recorded_at")
    else:
        session = getattr(client, "_claude_session", None)
        info = dict(getattr(session, "last_rate_limit", {}) or {}) if session is not None else {}
    return build_claude_code_usage_snapshot(rate_limit=info, recorded_at=recorded_at, note=note)


def fetch_account_usage(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto", "custom"}:
        return None
    try:
        if normalized == "openai-codex":
            return _fetch_codex_account_usage(base_url=base_url, api_key=api_key)
        if normalized == "anthropic":
            return _fetch_anthropic_account_usage()
        if normalized == "openrouter":
            return _fetch_openrouter_account_usage(base_url, api_key)
        if normalized == "claude-code":
            # The CLI's /usage waits for this call (its executor joins on
            # exit), so a hung control process must not hold it for long.
            return fetch_claude_code_account_usage(timeout=10.0)
    except Exception:
        return None
    return None
