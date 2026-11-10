"""Exercise the CLI through LiteLLM and a loopback Chat Completions server."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from trpg_agents.cli import main
from trpg_agents.config import HOSTED_API_BASES, HOSTED_LITELLM_PROVIDERS, ModelConfig


@pytest.fixture
def endpoint():
    """Serve a predictable reply and record the request path, headers and body."""
    requests = []
    reply = {
        "status": 200,
        "headers": [],
        "body": {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "vendor/test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        },
    }

    class Handler(BaseHTTPRequestHandler):
        """Capture one HTTP request and return the fixture's response."""

        def do_POST(self):
            """Read the chat payload and record any Authorization header."""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            reply["headers"].append(dict(self.headers))
            content = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            if reply.get("location"):
                self.send_header("Location", reply["location"])
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/custom/v1", requests, reply
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("requires_key", [False, True])
def test_connection_uses_the_configured_endpoint_and_key(
    endpoint, tmp_path, monkeypatch, capsys, requires_key
):
    """Send only the selected credential and preserve the engine's model ID."""
    base_url, requests, _ = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-cloud-key")
    monkeypatch.setenv("TRPG_TEST_KEY", "endpoint-test-key")
    config = tmp_path / "llm.toml"
    key_setting = '\napi_key_env = "TRPG_TEST_KEY"' if requires_key else ""
    config.write_text(f'base_url = "{base_url}"\nmodel = "vendor/test-model"{key_setting}\n')

    assert main(["test-connection", "--config", str(config)]) == 0
    assert capsys.readouterr().out.strip() == "Connection OK: vendor/test-model"
    assert len(requests) == 1
    path, auth, payload = requests[0]
    assert path == "/custom/v1/chat/completions"
    assert auth == ("Bearer endpoint-test-key" if requires_key else None)
    assert payload["model"] == "vendor/test-model"
    assert payload["messages"] == [{"role": "user", "content": "Reply with one word: OK."}]


def test_connection_reports_auth_failure_without_response_body(
    endpoint, tmp_path, monkeypatch, capsys
):
    """Report the HTTP status while leaving the server's error body out of output."""
    base_url, requests, reply = endpoint
    reply.update(status=401, body={"error": {"message": "server-echoed-sensitive-text"}})
    config = tmp_path / "llm.toml"
    config.write_text(f'base_url = "{base_url}"\nmodel = "test-model"\napi_key_env = "TEST_KEY"\n')
    monkeypatch.setenv("TEST_KEY", "endpoint-test-key")

    assert main(["test-connection", "--config", str(config)]) == 1
    output = capsys.readouterr()
    assert "HTTP 401" in output.err
    assert "server-echoed-sensitive-text" not in output.out + output.err
    assert "endpoint-test-key" not in output.out + output.err
    assert len(requests) == 1


def test_connection_with_missing_key_fails_before_sending(endpoint, tmp_path, monkeypatch, capsys):
    """Treat a missing configured credential as a local configuration error."""
    base_url, requests, _ = endpoint
    config = tmp_path / "llm.toml"
    config.write_text(f'base_url = "{base_url}"\nmodel = "test-model"\napi_key_env = "TEST_KEY"\n')
    monkeypatch.delenv("TEST_KEY", raising=False)

    assert main(["test-connection", "--config", str(config)]) == 1
    assert "Set TEST_KEY" in capsys.readouterr().err
    assert requests == []


@pytest.mark.parametrize("provider", list(HOSTED_API_BASES))
def test_hosted_provider_selects_native_route(provider, tmp_path, monkeypatch, capsys):
    """Pass the exact model and selected key to LiteLLM's native provider route."""
    import litellm

    calls = []

    def fake_completion(**kwargs):
        """Capture LiteLLM's route arguments without sending a paid request."""
        calls.append(kwargs)
        return ModelResponse(choices=[Choices(index=0, message=Message(content="OK"))])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://wrong.invalid")
    monkeypatch.setenv("GEMINI_API_BASE", "https://wrong.invalid")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://wrong.invalid")
    config = tmp_path / "hosted.toml"
    config.write_text(
        f'provider = "{provider}"\nmodel = "vendor/exact-id"\napi_key_env = "TEST_HOSTED_KEY"\n'
    )

    assert main(["test-connection", "--config", str(config)]) == 0
    assert capsys.readouterr().out.strip() == "Connection OK: vendor/exact-id"
    assert len(calls) == 1
    assert calls[0]["custom_llm_provider"] == HOSTED_LITELLM_PROVIDERS.get(provider, provider)
    assert calls[0]["model"] == "vendor/exact-id"
    assert calls[0]["api_base"] == HOSTED_API_BASES[provider]
    assert calls[0]["api_key"] == "selected-key"


def test_config_modes_are_exclusive_and_hosted_key_name_is_required():
    """Accept a hosted config without a base URL and reject ambiguous settings."""
    hosted = ModelConfig(model="test-model", provider="anthropic", api_key_env="TEST_KEY")
    assert hosted.base_url is None
    with pytest.raises(ValueError, match="exactly one"):
        ModelConfig(base_url="https://example.invalid/v1", model="test-model", provider="openai")
    with pytest.raises(ValueError, match="require api_key_env"):
        ModelConfig(model="test-model", provider="openai")
    with pytest.raises(ValueError, match="provider must be"):
        ModelConfig(model="test-model", provider="not-a-provider", api_key_env="TEST_KEY")


@pytest.mark.parametrize("model", ["gpt-5-pro", "responses/alias", "openai/vendor/model"])
@pytest.mark.parametrize("requires_key", [False, True])
def test_custom_endpoint_keeps_chat_protocol_and_isolates_ambient_headers(
    endpoint, tmp_path, monkeypatch, capsys, model, requires_key
):
    """Keep literal model IDs and selected authentication on the actual request."""
    base_url, requests, reply = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "Authorization: Bearer ambient-custom-key\nX-Ambient-Secret: leaked",
    )
    monkeypatch.setenv("TRPG_SELECTED_KEY", "selected-key")
    config = tmp_path / "custom.toml"
    key_setting = '\napi_key_env = "TRPG_SELECTED_KEY"' if requires_key else ""
    config.write_text(f'base_url = "{base_url}"\nmodel = "{model}"{key_setting}\n')

    assert main(["test-connection", "--config", str(config)]) == 0
    assert capsys.readouterr().out.strip() == f"Connection OK: {model}"
    assert len(requests) == 1
    path, authorization, body = requests[0]
    assert path == "/custom/v1/chat/completions"
    assert body["model"] == model
    assert authorization == ("Bearer selected-key" if requires_key else None)
    headers = {name.lower(): value for name, value in reply["headers"][0].items()}
    assert "x-ambient-secret" not in headers
    assert "openai-organization" not in headers
    assert "openai-project" not in headers


@pytest.mark.parametrize("provider", [None, "openai", "qwen", "kimi", "glm"])
def test_experimental_openai_handler_fails_before_sending(
    endpoint, tmp_path, monkeypatch, capsys, provider
):
    """Reject an ambient handler that bypasses the isolated OpenAI client."""
    base_url, requests, _ = endpoint
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "True")
    monkeypatch.setenv("TRPG_SELECTED_KEY", "selected-key")
    config = tmp_path / "model.toml"
    if provider is not None:
        config.write_text(
            f'provider = "{provider}"\nmodel = "test-model"\napi_key_env = "TRPG_SELECTED_KEY"\n'
        )
    else:
        config.write_text(
            f'base_url = "{base_url}"\nmodel = "test-model"\napi_key_env = "TRPG_SELECTED_KEY"\n'
        )

    assert main(["test-connection", "--config", str(config)]) == 1
    assert "Experimental OpenAI HTTP handler" in capsys.readouterr().err
    assert requests == []


@pytest.mark.parametrize("provider", [None, *HOSTED_API_BASES])
@pytest.mark.parametrize("proxy_source", ["environment", "sdk_global"])
def test_global_litellm_proxy_mode_fails_before_sending(
    endpoint, tmp_path, monkeypatch, capsys, provider, proxy_source
):
    """Reject LiteLLM's global proxy override for custom and hosted models."""
    import litellm

    base_url, requests, _ = endpoint
    monkeypatch.setenv("TRPG_SELECTED_KEY", "selected-key")
    if proxy_source == "environment":
        monkeypatch.setenv("USE_LITELLM_PROXY", "True")
        monkeypatch.setattr(litellm, "use_litellm_proxy", False)
    else:
        monkeypatch.delenv("USE_LITELLM_PROXY", raising=False)
        monkeypatch.setattr(litellm, "use_litellm_proxy", True)
    config = tmp_path / "model.toml"
    if provider is None:
        config.write_text(f'base_url = "{base_url}"\nmodel = "vendor/model"\n')
    else:
        config.write_text(
            f'provider = "{provider}"\nmodel = "vendor/model"\napi_key_env = "TRPG_SELECTED_KEY"\n'
        )

    assert main(["test-connection", "--config", str(config)]) == 1
    output = capsys.readouterr()
    assert "Global LiteLLM proxy mode" in output.err
    assert "selected-key" not in output.out + output.err
    assert requests == []


@pytest.mark.parametrize("status", [401, 422])
def test_custom_endpoint_reports_actual_http_status(endpoint, tmp_path, capsys, status):
    """Label a server response status without printing its body."""
    base_url, _, reply = endpoint
    reply.update(status=status, body={"error": {"message": "private-response-body"}})
    config = tmp_path / "custom.toml"
    config.write_text(f'base_url = "{base_url}"\nmodel = "test-model"\n')

    assert main(["test-connection", "--config", str(config)]) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert f"HTTP {status}" in output
    assert "private-response-body" not in output


def test_malformed_success_does_not_claim_http_500(endpoint, tmp_path, capsys):
    """Avoid presenting a synthesized SDK error as a server HTTP status."""
    base_url, _, reply = endpoint
    reply["body"] = {"unexpected": "private-response-body"}
    config = tmp_path / "custom.toml"
    config.write_text(f'base_url = "{base_url}"\nmodel = "test-model"\n')

    assert main(["test-connection", "--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert "HTTP 200" in captured.err
    assert "HTTP 500" not in captured.err
    assert "private-response-body" not in captured.out + captured.err


@pytest.mark.parametrize(
    "provider,expected_path",
    [
        ("anthropic", "/v1/messages"),
        ("gemini", "/v1beta/models/gemini-2.5-flash:generateContent"),
    ],
)
def test_non_openai_hosted_provider_uses_native_wire_protocol(
    endpoint, tmp_path, monkeypatch, capsys, provider, expected_path
):
    """Inspect a synthetic native-provider request without reaching a cloud API."""
    base_url, requests, reply = endpoint
    test_api_base = base_url.removesuffix("/custom/v1")
    if provider == "gemini":
        test_api_base += "/v1beta"
    monkeypatch.setitem(HOSTED_API_BASES, provider, test_api_base)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    config = tmp_path / "hosted.toml"
    config.write_text(
        f'provider = "{provider}"\nmodel = "gemini-2.5-flash"\napi_key_env = "TEST_HOSTED_KEY"\n'
    )
    main(["test-connection", "--config", str(config)])
    capsys.readouterr()
    assert len(requests) == 1
    assert requests[0][0] == expected_path
    headers = {name.lower(): value for name, value in reply["headers"][0].items()}
    if provider == "anthropic":
        assert headers["x-api-key"] == "selected-key"
    else:
        assert headers["x-goog-api-key"] == "selected-key"


def test_hosted_openai_responses_bridge_keeps_selected_key(endpoint, tmp_path, monkeypatch, capsys):
    """Inspect the native Responses bridge's headers with ambient OpenAI settings."""
    base_url, requests, reply = endpoint
    monkeypatch.setitem(HOSTED_API_BASES, "openai", base_url)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-key\nX-Ambient: secret"
    )
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    config = tmp_path / "hosted.toml"
    config.write_text('provider = "openai"\nmodel = "gpt-5-pro"\napi_key_env = "TEST_HOSTED_KEY"\n')

    main(["test-connection", "--config", str(config)])
    capsys.readouterr()
    assert len(requests) == 1
    assert requests[0][0] == "/custom/v1/responses"
    assert requests[0][1] == "Bearer selected-key"
    headers = {name.lower(): value for name, value in reply["headers"][0].items()}
    assert "x-ambient" not in headers
    assert "openai-organization" not in headers
    assert "openai-project" not in headers


def test_hosted_openrouter_preserves_native_model_and_selected_key(
    endpoint, tmp_path, monkeypatch, capsys
):
    """Send an OpenRouter provider/model ID through its Chat Completions route."""
    base_url, requests, reply = endpoint
    monkeypatch.setitem(HOSTED_API_BASES, "openrouter", base_url)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-router-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-key")
    config = tmp_path / "hosted.toml"
    config.write_text(
        'provider = "openrouter"\nmodel = "anthropic/claude-test"\n'
        'api_key_env = "TEST_HOSTED_KEY"\n'
    )

    assert main(["test-connection", "--config", str(config)]) == 0
    capsys.readouterr()
    assert len(requests) == 1
    assert requests[0][0] == "/custom/v1/chat/completions"
    assert requests[0][1] == "Bearer selected-key"
    assert requests[0][2]["model"] == "anthropic/claude-test"
    headers = {name.lower(): value for name, value in reply["headers"][0].items()}
    assert "x-ambient" not in headers


@pytest.mark.parametrize(
    "provider,model",
    [
        ("qwen", "qwen-plus"),
        ("kimi", "kimi-k2.5"),
        ("glm", "glm-4.6"),
        ("minimax", "MiniMax-M2.1"),
    ],
)
def test_china_family_hosted_routes_use_selected_key_and_native_wire(
    endpoint, tmp_path, monkeypatch, capsys, provider, model
):
    """Exercise the locked LiteLLM provider adapter against a local HTTP server."""
    base_url, requests, reply = endpoint
    monkeypatch.setitem(HOSTED_API_BASES, provider, base_url)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-key\nX-Ambient: secret"
    )
    for name in ("DASHSCOPE_API_KEY", "MOONSHOT_API_KEY", "ZAI_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.setenv(name, "ambient-provider-key")
    config = tmp_path / "hosted.toml"
    config.write_text(
        f'provider = "{provider}"\nmodel = "{model}"\napi_key_env = "TEST_HOSTED_KEY"\n'
    )

    assert main(["test-connection", "--config", str(config)]) == 0
    assert capsys.readouterr().out.strip() == f"Connection OK: {model}"
    assert len(requests) == 1
    path, authorization, body = requests[0]
    assert path == "/custom/v1/chat/completions"
    assert authorization == "Bearer selected-key"
    assert body["model"] == model
    assert body["messages"] == [{"role": "user", "content": "Reply with one word: OK."}]
    headers = {name.lower(): value for name, value in reply["headers"][0].items()}
    assert "x-ambient" not in headers
    assert "openai-organization" not in headers
    assert "openai-project" not in headers


@pytest.mark.parametrize("status", [429, 503])
def test_glm_retryable_http_failure_sends_once(endpoint, tmp_path, monkeypatch, capsys, status):
    """Limit both LiteLLM and its Z.AI SDK client to one outbound attempt."""
    base_url, requests, reply = endpoint
    reply.update(status=status, body={"error": {"message": "private-response-body"}})
    monkeypatch.setitem(HOSTED_API_BASES, "glm", base_url)
    monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
    config = tmp_path / "glm.toml"
    config.write_text('provider = "glm"\nmodel = "glm-4.6"\napi_key_env = "TEST_HOSTED_KEY"\n')

    assert main(["test-connection", "--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert "private-response-body" not in captured.out + captured.err
    assert len(requests) == 1
    path, authorization, body = requests[0]
    assert path == "/custom/v1/chat/completions"
    assert authorization == "Bearer selected-key"
    assert body["model"] == "glm-4.6"
    assert "max_retries" not in body


def test_custom_endpoint_does_not_follow_cross_origin_redirect(
    endpoint, tmp_path, monkeypatch, capsys
):
    """Keep the configured credential away from a second loopback origin."""
    base_url, requests, reply = endpoint
    redirected_requests = []

    class RedirectTarget(BaseHTTPRequestHandler):
        """Record requests sent to a different loopback origin."""

        def do_POST(self):
            """Record a redirected request if the client follows the Location header."""
            redirected_requests.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

    with ThreadingHTTPServer(("127.0.0.1", 0), RedirectTarget) as target:
        thread = Thread(target=target.serve_forever)
        thread.start()
        try:
            reply["status"] = 307
            reply["location"] = f"http://127.0.0.1:{target.server_port}/chat/completions"
            monkeypatch.setenv("TRPG_SELECTED_KEY", "selected-key")
            config = tmp_path / "custom.toml"
            config.write_text(
                f'base_url = "{base_url}"\nmodel = "test-model"\n'
                'api_key_env = "TRPG_SELECTED_KEY"\n'
            )
            assert main(["test-connection", "--config", str(config)]) == 1
        finally:
            target.shutdown()
            thread.join()
    captured = capsys.readouterr()
    assert "HTTP 307" in captured.err
    assert len(requests) == 1
    assert redirected_requests == []


@pytest.mark.parametrize("provider", ["qwen", "kimi", "glm", "minimax"])
def test_china_family_hosted_routes_do_not_follow_cross_origin_redirect(
    endpoint, tmp_path, monkeypatch, capsys, provider
):
    """Keep the selected provider key at its configured origin after HTTP 307."""
    base_url, requests, reply = endpoint
    redirected_requests = []

    class RedirectTarget(BaseHTTPRequestHandler):
        """Capture any request sent to another origin."""

        def do_POST(self):
            """Record a redirected request if the provider client follows it."""
            redirected_requests.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

    with ThreadingHTTPServer(("127.0.0.1", 0), RedirectTarget) as target:
        thread = Thread(target=target.serve_forever)
        thread.start()
        try:
            reply["status"] = 307
            reply["location"] = f"http://127.0.0.1:{target.server_port}/chat/completions"
            monkeypatch.setitem(HOSTED_API_BASES, provider, base_url)
            monkeypatch.setenv("TEST_HOSTED_KEY", "selected-key")
            config = tmp_path / "hosted.toml"
            config.write_text(
                f'provider = "{provider}"\nmodel = "test-model"\napi_key_env = "TEST_HOSTED_KEY"\n'
            )
            assert main(["test-connection", "--config", str(config)]) == 1
        finally:
            target.shutdown()
            thread.join()
    capsys.readouterr()
    assert len(requests) == 1
    assert redirected_requests == []
