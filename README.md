# trpg-agents

Shared model access for the Agentic TRPG agent layer. The Host sends requests
through LiteLLM to a selected hosted provider or a custom Chat Completions API.

## Test a model connection

1. Copy `llm.example.toml` to `llm.toml`. Set `provider` to `openai`, `anthropic`,
   `gemini`, `openrouter`, `qwen`, `kimi`, `glm`, or `minimax`, and `model` to the
   exact provider model ID. Set
   `api_key_env` to the **name** of a local environment variable holding the key.
   For example, in PowerShell 7:
   `$env:TRPG_LLM_API_KEY = Read-Host -MaskInput "Model API key"`.
2. For a custom server instead, replace `provider` with `base_url` (for example
   `http://127.0.0.1:8080/v1`) and use its exact Chat Completions model ID.
   `api_key_env` is optional for a custom server. The configured base path gets
   `/chat/completions` appended. Custom model IDs never select Responses.

Run from this checkout with Python 3.12 or newer and `uv`:

```powershell
uv run --locked trpg-agents test-connection --config llm.toml
```

The command sends a short prompt and exits with status 0 after receiving nonempty
reply text. A failure exits with status 1. The probe checks basic connectivity;
only synthetic local endpoints have been exercised so far.

The four additional built-in routes use fixed provider endpoints and LiteLLM's
provider implementations:

| `provider` | LiteLLM route | Platform and API base |
| --- | --- | --- |
| `qwen` | `dashscope` | Alibaba Cloud Model Studio, China (Beijing): `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `kimi` | `moonshot` | Moonshot China: `https://api.moonshot.cn/v1` |
| `glm` | `zai` | International Z.AI: `https://api.z.ai/api/paas/v4` |
| `minimax` | `minimax` | MiniMax China: `https://api.minimaxi.com/v1` |

Use credentials for the named platform. In particular, `glm` requires a key
issued by international Z.AI. The locked LiteLLM 1.108.4 provides the `zai`
route; domestic BigModel can be configured through its compatible endpoint in
custom mode. The four built-in routes send Chat Completions requests with the
provider model ID.
