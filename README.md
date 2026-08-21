# mittal-ai

Mittal Analytics' reusable AI harness. Install it as `mittal-ai` and import it as
`mittal_ai`. It provides:

- streaming, non-streaming and structured LLM responses;
- tool-call handling and message-history repair;
- token-cost calculation;
- OpenRouter routing and provider preferences;
- compatibility fixes for Chinese models; and
- `dj-evals` events for model requests and tool calls.

The application keeps its API keys. The model declares which provider and base URL
the harness should use:

```python
from mittal_ai import AIModel, get_client, get_structured_response
from pydantic import BaseModel


class Summary(BaseModel):
    text: str


model = AIModel(
    name="openai/gpt-5.4",
    api_key="...",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    input_tokens_cost_usd=2.5,
    input_tokens_cached_cost_usd=0.25,
    output_tokens_cost_usd=15,
    output_tokens_reasoning_cost_usd=15,
)

async with get_client(model) as client:
    async for event in get_structured_response(
        client=client,
        ai_model=model,
        input=[{"role": "user", "content": "Summarise this."}],
        tools=[],
        text_format=Summary,
        reasoning_effort="low",
    ):
        print(event)
```

The main public entry points are `get_response`, `get_streaming_response` and
`get_structured_response`.

## Publishing a new version

The [release workflow](.github/workflows/release.yml) publishes version tags to
PyPI using trusted publishing, so it does not store a PyPI API token. Before
starting a release, commit all changes that should be included and make sure the
working tree is clean.

For the next patch release:

```bash
uv version --bump patch
version=$(uv version --short)

uv run pytest
uv run pre-commit run --all-files

git add pyproject.toml uv.lock
git commit -m "Release version $version"
git push origin main

git tag -a "v$version" -m "v$version"
git push origin "v$version"
```

Pushing the tag starts the workflow. It runs the tests, builds and installs the
wheel and source distribution, generates attestations, and publishes the files
to PyPI.

Use `uv version --bump minor` or `uv version --bump major` when appropriate. The
tag must match the version in `pyproject.toml`, with a `v` prefix. PyPI release
versions are immutable, so every release needs a new version.
