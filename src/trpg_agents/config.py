"""Read hosted-provider or custom Chat Completions connection settings."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

HOSTED_API_BASES = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "glm": "https://api.z.ai/api/paas/v4",
    "minimax": "https://api.minimaxi.com/v1",
}

HOSTED_LITELLM_PROVIDERS = {
    "qwen": "dashscope",
    "kimi": "moonshot",
    "glm": "zai",
    "minimax": "minimax",
}


@dataclass(frozen=True)
class ModelConfig:
    """Identify a hosted LiteLLM provider or a custom Chat Completions endpoint."""

    base_url: str | None = None
    model: str = ""
    api_key_env: str | None = None
    provider: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid endpoint settings before the Host attempts a request."""
        if (self.base_url is None) == (self.provider is None):
            raise ValueError("Set exactly one of provider or base_url")
        if self.provider is not None:
            if not isinstance(self.provider, str) or self.provider not in HOSTED_API_BASES:
                raise ValueError(
                    "provider must be openai, anthropic, gemini, openrouter, "
                    "qwen, kimi, glm, or minimax"
                )
            if self.api_key_env is None:
                raise ValueError("Hosted providers require api_key_env")
        else:
            if not isinstance(self.base_url, str) or any(char.isspace() for char in self.base_url):
                raise ValueError("base_url must be an HTTP or HTTPS API base URL")
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname:
                raise ValueError("base_url must be an HTTP or HTTPS API base URL")
            if url.username is not None or url.query or url.fragment:
                raise ValueError("base_url must omit credentials, query parameters, and fragments")
            if url.port == 0:
                raise ValueError("base_url must name a reachable port")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty model ID")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str) or not self.api_key_env.isidentifier()
        ):
            raise ValueError("api_key_env must name an environment variable")

    def read_api_key(self) -> str | None:
        """Read only the configured secret and fail if it is missing or empty."""
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if not key or not key.strip():
            raise ValueError(f"Set {self.api_key_env} before connecting to this model")
        return key


def load_model_config(path: Path) -> ModelConfig:
    """Load a TOML model configuration without resolving its API key."""
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    if "model" not in settings:
        raise ValueError("Model config requires model")
    allowed = {"base_url", "provider", "model", "api_key_env"}
    if settings.keys() - allowed:
        raise ValueError("Model config accepts only provider, base_url, model, and api_key_env")
    return ModelConfig(
        base_url=settings.get("base_url"),
        model=settings["model"],
        api_key_env=settings.get("api_key_env"),
        provider=settings.get("provider"),
    )
