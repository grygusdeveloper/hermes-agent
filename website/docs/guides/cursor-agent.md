---
title: "Cursor Agent"
sidebar_label: "Cursor Agent"
---

# Cursor Agent provider

Hermes can use the official Cursor Agent CLI as a first-class provider. Cursor owns the browser-login credentials and refresh lifecycle; Hermes verifies status, discovers models, and launches Cursor's ACP stdio transport. No Cursor token is copied into `~/.hermes/auth.json`, repository files, logs, or process arguments.

## Requirements

1. Install Cursor Agent so `agent` is on `PATH`.
2. Have a Cursor account/plan entitled to the models you intend to use.
3. Confirm the model exists for your account with `agent models`.

A Cursor consumer subscription login is distinct from a direct xAI API key. The `cursor` provider consumes Cursor account entitlements through Cursor Agent. For direct xAI billing, use Hermes's `xai` or `xai-oauth` provider instead.

## Authenticate and select a model

```bash
hermes auth add cursor
hermes auth status cursor
hermes model
```

For a remote shell where a browser cannot open automatically:

```bash
hermes auth add cursor --no-browser
```

Hermes calls Cursor's supported `agent login` flow and then verifies `agent status --format json`. `hermes auth logout cursor` calls Cursor's supported logout and verifies that the session is gone.

Model selection uses live `agent models` discovery. Hermes passes the selected exact ID to Cursor as separate arguments, for example:

```text
agent --model cursor-grok-4.6-high acp
```

Credentials are not present in that command line.

## Non-default CLI location

Only the non-secret executable path is stored in Hermes configuration. Prefer the dedicated Cursor section; `providers.cursor.command` is also accepted:

```yaml
cursor:
  command: /absolute/path/to/agent
```

```yaml
providers:
  cursor:
    command: /absolute/path/to/agent
```

The value must name one executable, not a shell command or argument string.

## Current authentication scope

The first-class flow supports Cursor-owned subscription/browser authentication. Hermes deliberately does not import opaque Cursor session files and does not yet pool `CURSOR_API_KEY` or `CURSOR_AUTH_TOKEN` values. `hermes auth add cursor --type api-key` therefore fails with an actionable message rather than pretending to store a key.

## Troubleshooting

- **Cursor Agent not found:** install it or set `cursor.command`.
- **Not authenticated:** run `hermes auth add cursor`, then `hermes auth status cursor`.
- **Model missing:** run `agent models`; availability is controlled by Cursor and your account.
- **ACP launch fails:** run `agent --model <exact-id> acp` directly to confirm the local CLI supports that model and mode.
- **Need direct xAI usage:** select the `xai`/`xai-oauth` provider; Cursor and xAI authentication are separate.

The legacy `copilot-acp` provider remains GitHub Copilot-specific. Cursor is not represented as Copilot and does not require the `HERMES_COPILOT_ACP_COMMAND` compatibility override.
