import asyncio
from contextlib import aclosing
from typing import Any, AsyncIterator, cast

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.responses import Response, ResponseFunctionToolCall

from callable_ai import (
    AIModel,
    EvalEvent,
    ToolCallResult,
    get_response,
    get_streaming_response,
)
from callable_ai.responses import INTERRUPTED_TOOL_OUTPUT, _process_tool_calls


def _get_model() -> AIModel:
    return AIModel(
        name="test-model",
        api_key="secret",
        input_tokens_cost_usd=0,
        input_tokens_cached_cost_usd=0,
        output_tokens_cost_usd=0,
        output_tokens_reasoning_cost_usd=0,
    )


def _get_tool_call(
    name: str, arguments: str = "{}", *, index: int = 0
) -> ChoiceDeltaToolCall:
    return ChoiceDeltaToolCall(
        index=index,
        id=f"call-{index + 1}",
        type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


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
    monkeypatch.setattr("callable_ai.responses.get_client", lambda _model: client)

    async with aclosing(
        get_streaming_response(
            user="user-1",
            ai_model=_get_model(),
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            reasoning_effort=None,
        )
    ) as response:
        assert await anext(response) == "hello"

    assert stream.closed
    assert client.closed


async def test_chat_completion_normalizes_tool_calls_and_forwards_progress():
    async def answer_question(
        question: str,
    ) -> AsyncIterator[EvalEvent | ToolCallResult]:
        yield {"type": "dj_evals.event", "message": "Searching"}
        yield {"content": question}

    messages = []
    events = [
        event
        async for event in _process_tool_calls(
            messages=messages,
            tools=[answer_question],
            pending_tool_calls=[
                _get_tool_call("answer_question", '{"question":"hello"}')
            ],
            spend=0,
            max_spend=100,
        )
    ]

    assert events == [
        ResponseFunctionToolCall(
            arguments='{"question":"hello"}',
            call_id="call-1",
            name="answer_question",
            type="function_call",
        ),
        {"type": "dj_evals.event", "message": "Searching"},
    ]
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "hello",
            "annotations": [],
        }
    ]


async def test_closing_tool_call_event_repairs_every_pending_call():
    def answer():
        return {"content": "done"}

    messages = []
    tool_calls = [_get_tool_call("answer", index=index) for index in range(2)]

    async with aclosing(
        _process_tool_calls(
            messages=messages,
            tools=[answer],
            pending_tool_calls=tool_calls,
            spend=0,
            max_spend=100,
        )
    ) as events:
        await anext(events)

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": f"call-{index + 1}",
            "content": INTERRUPTED_TOOL_OUTPUT,
        }
        for index in range(2)
    ]


async def test_chat_completion_tool_failures_are_shown_and_returned_to_model():
    def fail():
        raise ValueError("broken")

    messages = []
    events = [
        event
        async for event in _process_tool_calls(
            messages=messages,
            tools=[fail],
            pending_tool_calls=[_get_tool_call("fail")],
            spend=0,
            max_spend=100,
        )
    ]

    assert isinstance(events[0], ResponseFunctionToolCall)
    assert events[1] == ResponseFunctionToolCall(
        arguments='{"msg": "Unable to process this tool call."}',
        call_id="call-1",
        name="gen_error",
        type="function_call",
    )
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "ValueError('broken')",
            "annotations": [],
        }
    ]


async def test_closing_response_cancels_tool_after_forwarded_event():
    cancelled = asyncio.Event()

    async def wait_forever(question: str):
        """Wait until cancelled.

        Args:
            - question: Question to answer.
        """
        try:
            yield {"type": "dj_evals.event", "message": question}
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    tool_call = ResponseFunctionToolCall(
        arguments='{"question":"hello"}',
        call_id="call-1",
        name="wait_forever",
        type="function_call",
    )
    response = Response.model_construct(output=[tool_call], usage=None)

    class MockResponses:
        async def create(self, **_kwargs):
            return response

    client = type("Client", (), {"responses": MockResponses()})()
    input = []

    async with aclosing(
        get_response(
            ai_model=_get_model(),
            input=input,
            tools=[wait_forever],
            client=cast(Any, client),
            reasoning_effort=None,
        )
    ) as events:
        assert await anext(events) == {
            "type": "dj_evals.event",
            "message": 'Tool call: wait_forever\n\n```json\n{"question":"hello"}\n```',
        }
        await anext(events)  # Event forwarded by the running tool.

    assert cancelled.is_set()
    assert input[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": INTERRUPTED_TOOL_OUTPUT,
    }
