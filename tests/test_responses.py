import asyncio
from contextlib import aclosing

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from mittal_ai import AIModel, get_streaming_response
from mittal_ai.responses import _call_tool_calls_with_events


class MockStream:
    def __init__(self):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        yield ChatCompletionChunk(
            id="response-1",
            created=0,
            choices=[Choice(index=0, delta=ChoiceDelta(content="hello"))],
            model="test-model",
            object="chat.completion.chunk",
        )


class MockCompletions:
    def __init__(self, stream):
        self.stream = stream

    async def create(self, **_kwargs):
        return self.stream


class MockClient:
    def __init__(self, stream):
        self.chat = type("Chat", (), {"completions": MockCompletions(stream)})()
        self.closed = False

    async def close(self):
        self.closed = True


async def test_closing_stream_stops_response_and_closes_client(monkeypatch):
    stream = MockStream()
    client = MockClient(stream)
    monkeypatch.setattr("mittal_ai.responses.get_client", lambda _model: client)
    model = AIModel(
        name="test-model",
        api_key="secret",
        input_tokens_cost_usd=0,
        input_tokens_cached_cost_usd=0,
        output_tokens_cost_usd=0,
        output_tokens_reasoning_cost_usd=0,
    )

    async with aclosing(
        get_streaming_response(
            user="user-1",
            ai_model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            reasoning_effort=None,
        )
    ) as response:
        assert await anext(response) == "hello"

    assert stream.closed
    assert client.closed


async def test_closing_tool_calls_cancels_pending_tools():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    tool_calls = _call_tool_calls_with_events(
        [(wait_forever, {})], spend=0, max_spend=100
    )
    consumer = asyncio.create_task(anext(tool_calls))
    await started.wait()

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert cancelled.is_set()
