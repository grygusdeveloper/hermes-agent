from agent.usage_pricing import get_pricing_entry, resolve_billing_route


def test_claude_code_is_subscription_included():
    route = resolve_billing_route("claude-opus-5", provider="claude-code", base_url="acp://claude-code")
    assert route.billing_mode == "subscription_included"
    entry = get_pricing_entry("claude-opus-5", provider="claude-code", base_url="acp://claude-code")
    assert entry is not None
    assert entry.pricing_version == "included-route"


def test_claude_code_marker_url_without_provider_is_subscription_included():
    route = resolve_billing_route("claude-sonnet-5", provider="", base_url="acp://claude-code")
    assert route.billing_mode == "subscription_included"
