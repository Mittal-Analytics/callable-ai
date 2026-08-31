# mittal-ai

Mittal Analytics' reusable AI harness. Install it as `mittal-ai` and import it as
`mittal_ai`. It provides:

- streaming, non-streaming and structured LLM responses;
- tool-call handling, reusable tool helpers and message-history repair;
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

## Tool helpers

Use `format_docstring` to customize a reusable tool description and
`partial_with_doc` to bind arguments that should not be exposed to the model:

```python
from mittal_ai import (
    ToolCallResult,
    format_docstring,
    get_streaming_response,
    partial_with_doc,
)


@format_docstring(entity="company")
def answer_question(_company_id: int, question: str) -> ToolCallResult:
    """Answer a question about a {entity}.

    Args:
        - question: Question asked by the user.
    """
    return {"content": f"{_company_id}: {question}"}


company_tool = partial_with_doc(answer_question, _company_id=123)

async for event in get_streaming_response(
    user="user-123",
    ai_model=model,
    messages=[{"role": "user", "content": "What changed?"}],
    tools=[company_tool],
    reasoning_effort="low",
):
    print(event)
```

`company_tool` exposes only `question`; its bound company ID stays private. Tools
return `ToolCallResult`; async generators may yield `EvalEvent` updates first.

## Publishing a new version

The [release workflow](.github/workflows/release.yml) publishes version tags to
PyPI using trusted publishing. From a clean working tree, publish the next patch
release with:

```bash
uv version --bump patch
version=$(uv version --short)

uv run pytest
uv run pre-commit run --all-files

git add pyproject.toml uv.lock
git commit -m "Release version $version"
git tag -a "v$version" -m "v$version"
git push origin main "v$version"
```

The tag must match the version in `pyproject.toml`, with a `v` prefix. Use
`--bump minor` or `--bump major` when appropriate. PyPI versions are immutable,
so every release needs a new version.
