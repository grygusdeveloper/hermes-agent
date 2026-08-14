"""GLM-5.3 catalog cutover tests for direct Z.AI and OpenCode Go."""

from hermes_cli.auth import ZAI_ENDPOINTS
from hermes_cli.models import _PROVIDER_MODELS
from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS


def test_zai_catalog_selects_glm_5_3_and_removes_glm_5_2():
    assert _PROVIDER_MODELS["zai"][0] == "glm-5.3"
    assert "glm-5.2" not in _PROVIDER_MODELS["zai"]
    assert _DEFAULT_PROVIDER_MODELS["zai"][0] == "glm-5.3"
    assert "glm-5.2" not in _DEFAULT_PROVIDER_MODELS["zai"]


def test_opencode_go_catalog_selects_glm_5_3_and_removes_glm_5_2():
    assert "glm-5.3" in _PROVIDER_MODELS["opencode-go"]
    assert "glm-5.2" not in _PROVIDER_MODELS["opencode-go"]


def test_coding_plan_endpoint_probes_glm_5_3_not_glm_5_2():
    coding_endpoints = [ep for ep in ZAI_ENDPOINTS if ep[0].startswith("coding-")]
    assert coding_endpoints
    for _ep_id, _base_url, models, _label in coding_endpoints:
        assert models[0] == "glm-5.3"
        assert "glm-5.2" not in models
