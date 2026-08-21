# maai

Mittal Analytics' reusable AI harness. It provides:

- streaming, non-streaming and structured LLM responses;
- tool-call handling and message-history repair;
- token-cost calculation;
- OpenRouter routing and provider preferences;
- compatibility fixes for Chinese models; and
- `dj-evals` events for model requests and tool calls.

The application keeps its API keys. The model declares which provider and base URL
the harness should use:

```python
from maai import AIModel, get_client, get_structured_response
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
