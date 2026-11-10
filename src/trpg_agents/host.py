"""Send chat requests through LiteLLM with per-connection credential boundaries."""

import os
from collections.abc import Sequence
from contextlib import closing

from .config import HOSTED_API_BASES, HOSTED_LITELLM_PROVIDERS, ModelConfig


class ChatRequestError(RuntimeError):
    """Report a request failure without exposing credentials or response bodies."""


def _isolated_client(base_url: str, key: str | None, timeout: float, actual_status: list[int]):
    """Build an SDK client that isolates credentials for OpenAI-compatible routes."""
    from openai import DefaultHttpxClient, OpenAI

    ambient_names = {"openai-organization", "openai-project"}
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").splitlines():
        name, _, _ = line.partition(":")
        if name.strip():
            ambient_names.add(name.strip().lower())

    def isolate_request(request) -> None:
        """Remove inherited headers and install only the configured bearer token."""
        for name in ambient_names | {"authorization"}:
            request.headers.pop(name, None)
        if key is not None:
            request.headers["Authorization"] = f"Bearer {key}"

    def observe_response(response) -> None:
        """Remember a status actually returned by the configured HTTP endpoint."""
        actual_status.append(response.status_code)

    return OpenAI(
        base_url=base_url,
        api_key=key if key is not None else "unused",
        organization="",
        project="",
        default_headers={"Authorization": f"Bearer {key}"} if key else {},
        http_client=DefaultHttpxClient(
            follow_redirects=False,
            event_hooks={"request": [isolate_request], "response": [observe_response]},
        ),
        timeout=timeout,
        max_retries=0,
    )


def complete(
    config: ModelConfig, messages: Sequence[dict[str, str]], *, timeout: float = 60
) -> str:
    """Send one non-streaming request and return nonempty text from the selected model."""
    key = config.read_api_key()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    from litellm import LiteLLMProxyChatConfig, completion

    if LiteLLMProxyChatConfig.should_use_litellm_proxy_by_default():
        raise ValueError("Global LiteLLM proxy mode is incompatible with this connection")

    if (
        config.provider is None or config.provider in {"openai", "qwen", "kimi", "glm"}
    ) and os.environ.get("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "").strip().lower() == "true":
        raise ValueError("Experimental OpenAI HTTP handler is incompatible with this connection")

    actual_status: list[int] = []
    try:
        if config.provider is not None:
            kwargs = {
                "model": config.model,
                "custom_llm_provider": HOSTED_LITELLM_PROVIDERS.get(
                    config.provider, config.provider
                ),
                "api_base": HOSTED_API_BASES[config.provider],
                "api_key": key,
            }
            if config.provider == "glm":
                # Z.AI's LiteLLM adapter otherwise drops the zero SDK retry count.
                kwargs["allowed_openai_params"] = ["max_retries"]
            if config.provider in {"openai", "qwen", "kimi", "glm"}:
                with _isolated_client(kwargs["api_base"], key, timeout, actual_status) as client:
                    response = completion(
                        **kwargs,
                        client=client,
                        messages=list(messages),
                        timeout=timeout,
                        num_retries=0,
                    )
            elif config.provider == "minimax":
                from litellm.llms.custom_httpx.http_handler import HTTPHandler

                with closing(HTTPHandler(timeout=timeout, follow_redirects=False)) as client:
                    response = completion(
                        **kwargs,
                        client=client,
                        messages=list(messages),
                        timeout=timeout,
                        num_retries=0,
                    )
            else:
                response = completion(
                    **kwargs, messages=list(messages), timeout=timeout, num_retries=0
                )
        else:
            base_url = config.base_url.rstrip("/")
            with _isolated_client(base_url, key, timeout, actual_status) as client:
                response = completion(
                    model="custom_openai/trpg-connection",
                    custom_llm_provider="custom_openai",
                    api_base=base_url,
                    api_key=key if key is not None else "unused",
                    client=client,
                    extra_body={"model": config.model},
                    messages=list(messages),
                    timeout=timeout,
                    num_retries=0,
                )
    except Exception as error:
        status = f", endpoint HTTP {actual_status[-1]}" if actual_status else ""
        raise ChatRequestError(f"Chat request failed ({type(error).__name__}{status})") from None

    text = response.choices[0].message.content if response.choices else None
    if not isinstance(text, str) or not text.strip():
        raise ChatRequestError("Chat endpoint returned no reply text")
    return text
