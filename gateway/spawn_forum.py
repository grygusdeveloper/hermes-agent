"""Bind native Discord forum posts to a brokered execution profile before their first turn."""

from __future__ import annotations

from gateway.config import Platform


async def bind_forum_post(runner, event) -> str | None:
    """Require exactly one configured model tag; persist model before the execution marker.

    A forum post is created with its first user message, so its model and credential
    scope must be durable before the gateway starts that message's agent turn.
    """
    source = event.source
    if source.platform != Platform.DISCORD or not source.thread_id or event.raw_message is None:
        return None
    from gateway.run import _load_gateway_config
    from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name

    raw = _load_gateway_config() or {}
    gateway = raw.get("gateway") if isinstance(raw, dict) else None
    spawn = gateway.get("spawn") if isinstance(gateway, dict) else None
    if not isinstance(spawn, dict) or not spawn.get("parent_channel_id"):
        return None
    if str(source.parent_chat_id or "") != str(spawn["parent_channel_id"]):
        return None

    entry = await runner.async_session_store.get_or_create_session(source)
    bound = await runner.async_session_store.get_execution_profile(entry.session_key)
    if bound:
        # /model may intentionally clear the override on a spawned lane. A
        # later post must not silently re-select its model from mutable tags.
        return None

    models = spawn.get("models")
    models = models if isinstance(models, dict) else {}
    channel = getattr(event.raw_message, "channel", None)
    tags = getattr(channel, "applied_tags", None) or ()
    selected = [alias for alias in models if any(
        str(getattr(tag, "name", "") or "").strip().casefold() == str(alias).strip().casefold()
        for tag in tags
    )]
    if len(selected) != 1:
        choices = ", ".join(f"`{alias}`" for alias in models) or "none"
        return f"❌ Select exactly one model tag for this forum post. Available: {choices}"
    alias = selected[0]
    spec = models[alias]
    if isinstance(spec, str):
        override = {"model": spec.strip()}
        mapped_profile = ""
    elif isinstance(spec, dict):
        override = {key: str(spec[key]).strip() for key in
                    ("model", "provider", "base_url", "reasoning_effort") if spec.get(key) not in (None, "")}
        mapped_profile = str(spec.get("profile") or "").strip()
    else:
        override, mapped_profile = {}, ""
    if not override.get("model"):
        return f"❌ Model tag `{alias}` has no configured model."

    agents = spawn.get("agents")
    agents = agents if isinstance(agents, dict) else {}
    default_agent = str(spawn.get("default_agent") or "main").strip().lower()
    agent_spec = agents.get(default_agent)
    if isinstance(agent_spec, str):
        profile = agent_spec
    elif isinstance(agent_spec, dict):
        profile = str(agent_spec.get("profile") or default_agent)
    else:
        profile = "default" if default_agent in {"main", "default"} else default_agent
    try:
        profile = normalize_profile_name(mapped_profile or profile)
        validate_profile_name(profile)
    except ValueError as exc:
        return f"❌ Model tag `{alias}` maps to an invalid profile: {exc}"
    if not profile_exists(profile):
        return f"❌ Model tag `{alias}` maps to missing profile `{profile}`."

    await runner.async_session_store.set_model_override(entry.session_key, override)
    await runner.async_session_store.set_execution_profile(entry.session_key, profile)
    runner._session_model_overrides.pop(entry.session_key, None)
    return None
