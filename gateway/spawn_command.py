"""Main-owned Discord workspace creation; execution binding is durable before kickoff."""

from __future__ import annotations

import dataclasses
import inspect
import re
import shlex

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import _prefix_within_utf16_limit, utf16_len
from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name

_STOPWORDS = {"a", "an", "and", "can", "could", "for", "help", "me", "please", "the", "to",
              "with", "would", "you", "czy", "dla", "mi", "możesz", "pomóż", "proszę", "w", "z"}
_TOPIC_SEPARATOR = " · "
_HEART_PREFIXES = ("❤️ ", "❤ ", "❤️", "❤")
_PLACEHOLDER_ESSENCES = {"new chat", "chat", "untitled", "test"}


def _split_heart(text: str) -> tuple[str, str]:
    """``(heart_prefix, rest)``; the prefix is normalized to ``"❤️ "`` so /heart state survives renames."""
    stripped = text.strip()
    prefix = next((p for p in _HEART_PREFIXES if stripped.startswith(p)), None)
    return ("❤️ ", stripped[len(prefix):].strip()) if prefix else ("", stripped)


def _is_topic_placeholder(name: str) -> bool:
    """True when a topic's essence (heart and `` · model`` suffix ignored) is empty or a stock placeholder
    such as ``.`` or ``New chat``: those topics wait for the session titler's semantic name."""
    essence = _split_heart(str(name or ""))[1].split(_TOPIC_SEPARATOR, 1)[0]
    cleaned = essence.strip(" .·-—_:;")
    return not cleaned or cleaned.casefold() in _PLACEHOLDER_ESSENCES


def _topic_title(text: str, label: str) -> str:
    """Keep the selected model visible even when Discord truncates a long title; a leading heart is kept."""
    heart, text = _split_heart(text.splitlines()[0] if text else "")
    words = re.findall(r"[^\W_][\w+.#'/-]*", re.sub(r"https?://\S+", " ", text), re.UNICODE)
    useful = [word for word in words if word.casefold() not in _STOPWORDS]
    essence = " ".join((useful or words)[:4])[:60] or "New chat"
    suffix = f"{_TOPIC_SEPARATOR}{_prefix_within_utf16_limit(label or 'Default', 75 - utf16_len(heart))}"
    essence = _prefix_within_utf16_limit(essence, max(1, 80 - utf16_len(heart) - utf16_len(suffix))).rstrip()
    return f"{heart}{essence or 'N'}{suffix}"


def _topic_model_label(config: dict, override: dict | None, tag_names: tuple[str, ...] = ()) -> str:
    """Friendly label of the session's model from the spawn catalog (then ``model_aliases``): the route
    whose set keys agree with the override. Sol High vs Sol XHigh share a wire model and the persisted
    override carries no effort, so ties go to the alias applied as the post's model tag."""
    model = str((override or {}).get("model") or "").strip()
    if not model:
        return "Default"
    gateway = config.get("gateway") if isinstance(config, dict) else None
    spawn = gateway.get("spawn") if isinstance(gateway, dict) else None
    spawn = spawn if isinstance(spawn, dict) else {}
    catalog = dict(spawn["models"]) if isinstance(spawn.get("models"), dict) else {}
    agents = spawn.get("agents") if isinstance(spawn.get("agents"), dict) else {}
    agent_spec = agents.get(str(spawn.get("default_agent") or "main").strip().lower())
    if isinstance(agent_spec, dict) and isinstance(agent_spec.get("models"), dict):
        catalog.update(agent_spec["models"])
    aliases = config.get("model_aliases") if isinstance(config, dict) else None
    tags = {str(name).strip().casefold() for name in tag_names}
    for section in (catalog, aliases if isinstance(aliases, dict) else {}):
        matches = {}
        for alias, spec in section.items():
            route = {"model": spec} if isinstance(spec, str) else spec
            if not isinstance(route, dict) or str(route.get("model") or "").strip() != model:
                continue
            if any(route.get(key) not in (None, "") and override.get(key) not in (None, "")
                   and str(route[key]).strip() != str(override[key]).strip()
                   for key in ("provider", "base_url", "reasoning_effort")):
                continue
            matches[str(alias)] = str(route.get("label") or alias).strip() or str(alias)
        for candidates in (matches, {a: label for a, label in matches.items() if a.strip().casefold() in tags}):
            if len(set(candidates.values())) == 1:
                return next(iter(candidates.values()))
    return model


def _parse_args(raw: str) -> tuple[dict[str, str], str | None]:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return {}, f"❌ Invalid `/spawn` arguments: {exc}"
    values = {key: "" for key in ("agent", "model", "title", "prompt")}
    positional = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--agent", "--model", "--title", "--prompt"}:
            index += 1
            if index >= len(tokens):
                return {}, f"❌ `{token}` requires a value."
            values[token[2:]] = tokens[index].strip()
        else:
            positional.append(token)
        index += 1
    for key in ("agent", "model"):
        if not values[key] and positional:
            values[key] = positional.pop(0)
    if not values["title"]:
        values["title"] = " ".join(positional)
    return values, None


async def handle_spawn(runner, event: MessageEvent) -> str:
    """Create and bind a thread on the receiving bot, then admit its optional first task."""
    if event.source.platform != Platform.DISCORD:
        return "`/spawn` is available only on Discord."
    from gateway.run import _load_gateway_config

    config = _load_gateway_config() or {}
    gateway = config.get("gateway") if isinstance(config, dict) else None
    spawn = gateway.get("spawn") if isinstance(gateway, dict) else None
    if not isinstance(spawn, dict) or not spawn.get("parent_channel_id"):
        return "❌ Set `gateway.spawn.parent_channel_id` before using `/spawn`."
    values, error = _parse_args(event.get_command_args().strip())
    if error:
        return error
    agent = (values["agent"] or str(spawn.get("default_agent") or "main")).strip().lower()
    agents = spawn.get("agents") if isinstance(spawn.get("agents"), dict) else {}
    agent_spec = agents.get(agent)
    profile = (agent_spec if isinstance(agent_spec, str) else
               str(agent_spec.get("profile") or agent) if isinstance(agent_spec, dict) else
               "default" if agent in {"main", "default"} else agent)
    model_alias = (values["model"] or str(spawn.get("default_model") or "")).strip().lower()
    configured_models = spawn.get("models")
    models = dict(configured_models) if isinstance(configured_models, dict) else {}
    if isinstance(agent_spec, dict) and isinstance(agent_spec.get("models"), dict):
        models.update(agent_spec["models"])
    override = None
    matches = []
    if model_alias:
        matches = [(alias, spec) for alias, spec in models.items() if str(alias).casefold() == model_alias.casefold()]
        if len(matches) != 1:
            return f"❌ Unknown or ambiguous model alias `{model_alias}`. Available: {', '.join(map(str, models)) or 'none'}"
        spec = matches[0][1]
        if isinstance(spec, str):
            override = {"model": spec.strip()}
        elif isinstance(spec, dict):
            override = {key: str(spec[key]).strip() for key in
                        ("model", "provider", "base_url", "reasoning_effort") if spec.get(key) not in (None, "")}
            profile = str(spec.get("profile") or profile)
        if not override or not override.get("model"):
            return f"❌ Model alias `{model_alias}` has no configured model."
    try:
        profile = normalize_profile_name(profile)
        validate_profile_name(profile)
    except ValueError as exc:
        return f"❌ Invalid spawn profile: {exc}"
    if not profile_exists(profile):
        return f"❌ Spawn profile `{profile}` is not installed."
    # Never take an adapter from the runtime profile: intake authorization and
    # all outbound messages belong to the Discord bot that received this command.
    adapter = runner._delivery_adapter_for(event.source)
    create = getattr(adapter, "create_spawn_thread", None)
    if not callable(create):
        return "❌ Receiving Discord bot cannot create a workspace thread."
    label = model_alias or "Default"
    if model_alias:
        spec = matches[0][1]
        if isinstance(spec, dict):
            label = str(spec.get("label") or label).strip()
    title = _topic_title(values["title"] or values["prompt"] or agent, label)
    parent = str(spawn["parent_channel_id"])
    result = await create(parent_chat_id=parent, name=title,
                          starter_content=f"🧵 Hermes workspace · agent `{agent}` · model `{model_alias or 'Default'}`",
                          owner_user_id=event.source.user_id, model_alias=model_alias or None)
    if not isinstance(result, dict) or not result.get("success"):
        return f"❌ Failed to create spawned thread: {result.get('error') if isinstance(result, dict) else 'unknown error'}"
    thread_id = str(result.get("thread_id") or "")
    if not thread_id:
        return "❌ Discord did not return a new thread ID."
    guild_id = str(result.get("guild_id") or event.source.guild_id or "") or None
    source = dataclasses.replace(event.source, chat_id=thread_id, chat_name=str(result.get("thread_name") or title),
                                 chat_type="thread", thread_id=thread_id, parent_chat_id=parent,
                                 scope_id=guild_id, guild_id=guild_id, message_id=None, profile=None)
    # dataclasses.replace intentionally does not copy an identity pinned for the
    # command's original channel; retain only the receiving adapter provenance.
    source._transport_adapter_ref = getattr(event.source, "_transport_adapter_ref", None)
    entry = await runner.async_session_store.get_or_create_session(source)
    if override:
        await runner.async_session_store.set_model_override(entry.session_key, override)
        runner._session_model_overrides.pop(entry.session_key, None)
    await runner.async_session_store.set_execution_profile(entry.session_key, profile)
    prompt = values["prompt"].strip()
    if prompt:
        kickoff = MessageEvent(text=prompt, message_type=MessageType.TEXT, source=source,
                               channel_prompt=event.channel_prompt)
        dispatch = adapter.handle_message(kickoff)
        if inspect.isawaitable(dispatch):
            await dispatch
        if not kickoff._gateway_accepted:
            return (f"⚠️ Spawned <#{thread_id}> · agent `{agent}` · model `{model_alias or 'Default'}` "
                    "but the initial task was not admitted. Please resend it in the thread.")
    return f"✅ Spawned <#{thread_id}> · agent `{agent}` · model `{model_alias or 'Default'}`" + (" · initial task queued" if prompt else "")
