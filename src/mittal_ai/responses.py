import asyncio
import inspect
import json
import logging
from collections import defaultdict
from contextlib import aclosing
from enum import Enum
from itertools import groupby
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

from dj_evals import EvalEvent
from django.core.exceptions import ObjectDoesNotExist
from openai import AsyncOpenAI, AsyncStream, BadRequestError, Omit, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    FunctionToolParam,
    ParsedResponse,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from openai.types.responses.response_output_text_param import (
    AnnotationURLCitation,
)
from pydantic import BaseModel, ValidationError

from mittal_ai import compatibility
from mittal_ai.costing import get_abs_cost, parse_responses_usage
from mittal_ai.models import AIModel
from mittal_ai.openrouter import (
    OpenRouterChatCompletionChunk,
    OpenRouterCompletionUsage,
    OpenRouterReasoningDetail,
    ReasoningDetailEncryptedType,
    ReasoningDetailSummaryType,
    ReasoningDetailTextType,
)

logger = logging.getLogger(__name__)


# Types
class UsageDetails(TypedDict):
    id: str
    model: str
    usage: OpenRouterCompletionUsage
    cost: float
    number_of_web_searches: NotRequired[int]


ParsedToolCallInfo: TypeAlias = Tuple[str, Dict[str, Any]]
StreamingResponseChunk: TypeAlias = (
    str
    | UsageDetails
    | ReasoningDetailSummaryType
    | ReasoningDetailTextType
    | ParsedToolCallInfo
)

Messages: TypeAlias = List[ChatCompletionMessageParam]


class OpenRouterAssistantMessageParam(ChatCompletionAssistantMessageParam):
    reasoning_details: NotRequired[List[OpenRouterReasoningDetail]]


INTERRUPTED_TOOL_OUTPUT = "InterruptedError: process was cancelled"


class AnnotationToolCallURL(TypedDict):
    type: Literal["tool_call_url"]
    title: str
    url: str


# https://platform.openai.com/docs/api-reference/chat/object
class ToolCallResult(TypedDict):
    usage_details: NotRequired[UsageDetails]
    annotations: NotRequired[List[AnnotationURLCitation | AnnotationToolCallURL]]
    content: str


# Keeps the parsed response type tied to the Pydantic schema passed in.
PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


AI_RESPONSE_TIMEOUT_SECONDS = 300


def _get_responses_options(options: Dict) -> Dict:
    reasoning_effort = options.pop("reasoning_effort", None)
    if reasoning_effort:
        # Responses API expects reasoning config under the `reasoning` key.
        options["reasoning"] = {"effort": reasoning_effort}

    # Needed for stateless multi-turn Responses API calls with store=False.
    options["include"] = ["reasoning.encrypted_content"]
    return options


def get_client(model: AIModel) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=model.api_key,
        timeout=AI_RESPONSE_TIMEOUT_SECONDS,
        base_url=model.base_url,
    )


def get_model_options(model: AIModel, reasoning_effort: Optional[str]) -> Dict:
    options = defaultdict(dict)
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    if model.extra_headers is not None:
        options["extra_headers"] = model.extra_headers

    if model.provider == "openrouter":
        # https://openrouter.ai/docs/features/message-transforms
        options["extra_body"]["transforms"] = ["middle-out"]
        if model.openrouter_providers:
            # Some OpenRouter models need a specific provider for stable routing.
            options["extra_body"]["provider"] = {"only": model.openrouter_providers}

        if "reasoning_effort" in options:
            # open-router supports passing `efforts` attribute in completions api
            # this is otherwise available only in responses api
            # https://platform.openai.com/docs/guides/reasoning/advice-on-prompting#reasoning-summaries
            options["extra_body"]["reasoning"] = {
                "effort": options.pop("reasoning_effort"),
                "summary": "concise",
            }
    return options


def _parse_docs(docs):
    # the __doc__ "strip a uniform amount of indentation from the second and further lines of the docstring, equal to the minimum indentation of all non-blank lines after the first line."
    # hence we are splitting at `\nArgs:\n`
    function_description, args_description = docs.strip().split("\nArgs:\n")
    function_description = function_description.strip()

    args_lines = args_description.split("\n    - ")
    arguments = {}
    for arg_line in args_lines:
        name, description = arg_line.split(": ", maxsplit=1)
        description = "\n".join(line.strip() for line in description.splitlines())
        arguments[name.strip(" -")] = description.strip()

    return {"function_docs": function_description, "arguments": arguments}


def _get_tools_definition(function: Callable) -> ChatCompletionFunctionToolParam:
    sig = inspect.signature(function)
    description = _parse_docs(function.__doc__)
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    for param_name, param in sig.parameters.items():
        if param_name.startswith("_"):
            continue

        other_param_properties = {}
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is str:
            param_type = "string"
        elif issubclass(param.annotation, Enum):
            param_type = "string"
            enum_values = [e.value for e in param.annotation]
            other_param_properties["enum"] = enum_values
        else:
            raise ValueError("undefined type", param.annotation)

        # https://platform.openai.com/docs/guides/function-calling?api-mode=chat&example=search-knowledge-base#handling-function-calls
        if param_name not in description["arguments"]:
            raise ValueError(
                "Documentation not provided in",
                param_name,
                function.__name__,  # type: ignore
            )
        param_properties = {
            "type": param_type,
            "description": description["arguments"][param_name],
            **other_param_properties,
        }

        # all properties must be marked as required in strict mode
        # https://platform.openai.com/docs/guides/function-calling#strict-mode
        # optional fields should take `null` as type option
        parameters["required"].append(param_name)
        if param.default != inspect._empty:
            raise ValueError(
                "defaults not implemented by us. underscore the param to skip it.",
                function,
                param,
            )

        parameters["properties"][param_name] = param_properties

    tool_definition = ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name=getattr(function, "__name__"),
            description=description["function_docs"],
            parameters=cast(Dict[str, object], parameters),
            strict=True,
        ),
    )
    return tool_definition


def _get_response_tools_definition(tool: Callable) -> FunctionToolParam:
    tool_def = _get_tools_definition(tool)
    return FunctionToolParam(
        type="function",
        name=tool_def["function"]["name"],
        parameters=tool_def["function"]["parameters"],
        strict=tool_def["function"]["strict"],
        description=tool_def["function"]["description"],
    )


async def gen_error(msg) -> ToolCallResult:
    logger.debug("gen error: %s", msg)
    return {"content": msg}


def _parse_tool_call(
    tools: Optional[List[Callable]],
    tool_call: ChatCompletionMessageFunctionToolCallParam,
) -> Tuple[Callable, Any]:
    available = {tool.__name__: tool for tool in tools} if tools else {}  # type: ignore
    name = tool_call["function"]["name"]
    if name not in available:
        function = gen_error
        kwargs = {"msg": f"Function name error - unknown name: {name}"}
    else:
        function = available[name]
        try:
            kwargs = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            function = gen_error
            kwargs = {"msg": "Couldn't parse the arguments to the tool"}

    return function, kwargs


def _gather_reasoning_details_chunks(
    reasoning_detail_chunks: List[OpenRouterReasoningDetail],
):
    def get_key(chunk):
        key = {"type": chunk["type"], "format": chunk["format"]}
        if "id" in chunk:
            key["id"] = chunk["id"]
        if "index" in chunk:
            key["index"] = chunk["index"]
        return key

    # https://openrouter.ai/docs/use-cases/reasoning-tokens#response-examples
    grouped: List[OpenRouterReasoningDetail] = []
    for key, items in groupby(reasoning_detail_chunks, key=get_key):
        items = list(items)
        if key["type"] == "reasoning.summary":
            summaries = cast(List[ReasoningDetailSummaryType], items)
            grouped.append(
                {**key, "summary": "".join(item["summary"] for item in summaries)}
            )
        elif key["type"] == "reasoning.text":
            texts = cast(List[ReasoningDetailTextType], items)
            grouped.append(
                {
                    **key,
                    "text": "".join(item["text"] for item in texts),
                    "signature": texts[0].get("signature"),
                }
            )
        else:
            grouped.extend(cast(List[ReasoningDetailEncryptedType], items))
    return grouped


def _gather_tool_call_chunks(
    tool_call_chunks: List[ChoiceDeltaToolCall],
) -> List[ChoiceDeltaToolCall]:
    # in case of gemini, the function calls are complete
    # the index is None in case of gemini, hence no accumulation required
    has_no_index = tool_call_chunks[0].index is None
    if has_no_index:
        return tool_call_chunks

    # handle function calls in streaming
    # https://platform.openai.com/docs/guides/function-calling?api-mode=chat#streaming
    #
    # in case of openai, the arguments of the function calls are also streamed
    # hence they need to be accumulated
    # the code below is from their page itself
    final_tool_calls = {}
    for tool_call in tool_call_chunks:
        if tool_call.index not in final_tool_calls:
            final_tool_calls[tool_call.index] = tool_call
        else:
            final_function = final_tool_calls[tool_call.index].function
            chunk_function = tool_call.function
            # Some providers send function metadata after the first tool-call chunk.
            if not chunk_function:
                continue
            if not final_function:
                final_tool_calls[tool_call.index].function = chunk_function
                continue
            if final_function.arguments is None:
                final_function.arguments = ""
            final_function.arguments += chunk_function.arguments or ""
    return list(final_tool_calls.values())


async def _call_tool_calls(
    tool_calls: List[Tuple[Callable, Any]], *, spend, max_spend
) -> List[ToolCallResult | BaseException]:
    tasks = []
    for function, kwargs in tool_calls:
        if spend >= max_spend:
            tasks.append(
                gen_error(
                    f"The amount spent on this answer has exceeded the maximum limit of ₹{max_spend}. The system didn't execute this tool call. Ask the user if they are okay to spend more. You can then call this tool call again if the user approves."
                )
            )
        else:
            try:
                tasks.append(function(**kwargs))
            except TypeError as e:
                tasks.append(gen_error(f"error: {e.args[0]}"))
            except ObjectDoesNotExist:
                tasks.append(gen_error("error: document not found"))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def _run_tool_call_with_events(
    index: int,
    function: Callable,
    kwargs: Any,
    queue: asyncio.Queue,
    *,
    spend,
    max_spend,
):
    if spend >= max_spend:
        result = await gen_error(
            f"The amount spent on this answer has exceeded the maximum limit of ₹{max_spend}. The system didn't execute this tool call. Ask the user if they are okay to spend more. You can then call this tool call again if the user approves."
        )
        await queue.put(("result", index, result))
        return

    try:
        result = function(**kwargs)
    except TypeError as e:
        result = await gen_error(f"error: {e.args[0]}")
    except ObjectDoesNotExist:
        result = await gen_error("error: document not found")
    except Exception as e:
        result = e
    else:
        try:
            if inspect.isasyncgen(result):
                tool_result = None
                async for event in result:
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "dj_evals.event"
                    ):
                        await queue.put(("event", event))
                    else:
                        tool_result = event
                if tool_result is None:
                    # Eval events are only for UI; the model still needs a tool result.
                    raise ValueError("Tool generator did not yield a result")
                result = tool_result
            elif inspect.isawaitable(result):
                result = await result
        except ObjectDoesNotExist:
            result = await gen_error("error: document not found")
        except Exception as e:
            result = e

    await queue.put(("result", index, result))


async def _call_tool_calls_with_events(
    tool_calls: List[Tuple[Callable, Any]], *, spend, max_spend
) -> AsyncIterator[EvalEvent | List[ToolCallResult | BaseException]]:
    queue = asyncio.Queue()
    tasks = []
    for index, (function, kwargs) in enumerate(tool_calls):
        # create_task schedules the coroutine immediately; it starts running when
        # this coroutine next awaits, which is queue.get() below.
        task = asyncio.create_task(
            _run_tool_call_with_events(
                index,
                function,
                kwargs,
                queue,
                spend=spend,
                max_spend=max_spend,
            )
        )
        # Count completed tasks, not just results, so a broken task cannot
        # make us think all work finished before the task is actually done.
        task.add_done_callback(lambda _task: queue.put_nowait(("done",)))
        tasks.append(task)

    results: dict[int, ToolCallResult | BaseException] = {}
    completed_tasks = 0
    while completed_tasks < len(tasks):
        # queue.get() sleeps until a tool emits an event or finishes; this is not
        # a CPU-spinning loop.
        event = await queue.get()
        if event[0] == "event":
            yield event[1]
        elif event[0] == "result":
            _type, index, result = event
            results[index] = result
        elif event[0] == "done":
            completed_tasks += 1

    yield [results[index] for index in range(len(tool_calls))]


def _append_interrupted_tool_outputs(
    messages: Messages, tool_call_ids: List[str], completed_tool_call_ids: set[str]
):
    for tool_call_id in tool_call_ids:
        if tool_call_id in completed_tool_call_ids:
            continue
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": INTERRUPTED_TOOL_OUTPUT,
            }
        )


def _is_empty_assistant_message(message) -> bool:
    return message.get("role") == "assistant" and not any(
        message.get(key)
        for key in [
            "content",
            "tool_calls",
            "reasoning_details",
            "function_call",
            "refusal",
        ]
    )


def _remove_empty_assistant_messages(messages: Messages) -> bool:
    fixed = [
        message for message in messages if not _is_empty_assistant_message(message)
    ]
    is_changed = len(fixed) != len(messages)
    if is_changed:
        # Keep the same list object used by websocket state.
        messages[:] = fixed
    return is_changed


def _repair_missing_tool_outputs(messages: Messages) -> bool:
    fixed = []
    pending_tool_call_ids = []

    def add_missing_outputs():
        for tool_call_id in pending_tool_call_ids:
            fixed.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": INTERRUPTED_TOOL_OUTPUT,
                }
            )
        pending_tool_call_ids.clear()

    for message in messages:
        if pending_tool_call_ids and message["role"] != "tool":
            # Old interrupted histories can miss tool outputs before next turn.
            add_missing_outputs()

        fixed.append(message)
        if message["role"] == "assistant":
            pending_tool_call_ids = [
                tool_call["id"]
                for tool_call in message.get("tool_calls", [])
                if tool_call.get("id")
            ]
        elif message["role"] == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id in pending_tool_call_ids:
                pending_tool_call_ids.remove(tool_call_id)

    add_missing_outputs()

    is_changed = len(fixed) != len(messages)
    if is_changed:
        # Keep the same list object used by websocket state.
        messages[:] = fixed
    return is_changed


def repair_message_history(messages: Messages) -> bool:
    removed_empty = _remove_empty_assistant_messages(messages)
    fixed_tools = _repair_missing_tool_outputs(messages)
    return removed_empty or fixed_tools


async def _process_tool_calls(
    *,
    messages: Messages,
    tools: List[Callable],
    pending_tool_calls: List[ChoiceDeltaToolCall],
    spend: float,
    max_spend: float,
) -> AsyncGenerator[UsageDetails | ParsedToolCallInfo, None]:
    # None is the type accepted by asend(); yielded values use the first parameter.
    # handle function calls concurrently
    # https://platform.openai.com/docs/guides/function-calling?api-mode=chat#handling-function-calls

    tool_calls = []
    tool_call_ids = []
    completed_tool_call_ids = set()
    for tool_call in pending_tool_calls:
        function, kwargs = _parse_tool_call(tools, tool_call.to_dict())  # type: ignore
        tool_calls.append((function, kwargs))
        tool_call_ids.append(cast(str, tool_call.id))

    try:
        for function, kwargs in tool_calls:
            yield (function.__name__, kwargs)  # type: ignore

        # yield tool calls and create coroutine tasks
        results = await _call_tool_calls(tool_calls, spend=spend, max_spend=max_spend)

        # Add results to messages
        for tool_call_id, result in zip(tool_call_ids, results):
            if isinstance(result, BaseException):
                logger.exception(
                    "Error occurred during tool call execution", exc_info=result
                )
                content = repr(result)
                yield (gen_error.__name__, {"msg": content})
                annotations = []
            else:
                if result.get("usage_details"):
                    yield result["usage_details"]
                    spend += result["usage_details"]["cost"]
                content = result["content"]
                annotations = result.get("annotations") or []
            messages.append(
                {  # type: ignore
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                    "annotations": annotations,
                }
            )
            completed_tool_call_ids.add(tool_call_id)
    except (asyncio.CancelledError, GeneratorExit):
        _append_interrupted_tool_outputs(
            messages, tool_call_ids, completed_tool_call_ids
        )
        raise


async def get_streaming_response(
    *,
    # user param improves cache routing, especially with lots of requests
    # https://platform.openai.com/docs/guides/prompt-caching
    # https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids
    user: str,
    ai_model: AIModel,
    messages: Messages,
    tools: List[Callable],
    reasoning_effort: Optional[str],
    spend: float = 0,
    max_spend: float = 100,
) -> AsyncGenerator[StreamingResponseChunk, None]:
    client = get_client(ai_model)
    options = get_model_options(ai_model, reasoning_effort=reasoning_effort)
    tool_definitions = [_get_tools_definition(tool) for tool in tools]

    async def reply(
        repair_attempt: int = 1,
    ) -> AsyncGenerator[StreamingResponseChunk, None]:
        nonlocal spend

        try:
            stream_response = await client.chat.completions.create(
                model=ai_model.name,
                messages=messages,
                tools=tool_definitions,
                stream=True,
                stream_options={"include_usage": True},
                store=False,
                user=user,
                **options,
            )
        except BadRequestError:
            # Retry only when we could repair old broken histories.
            if repair_attempt >= 3 or not repair_message_history(messages):
                raise

            async with aclosing(reply(repair_attempt + 1)) as retry_stream:
                async for chunk in retry_stream:
                    yield chunk
            return

        accumulated_content = ""
        tool_call_chunks: List[ChoiceDeltaToolCall] = []
        reasoning_detail_chunks: List[OpenRouterReasoningDetail] = []
        async with stream_response as raw_stream:
            stream = cast(AsyncStream[OpenRouterChatCompletionChunk], raw_stream)
            async for chunk in stream:
                if chunk.usage:
                    usage = cast(
                        OpenRouterCompletionUsage,
                        chunk.usage.to_dict()
                        if isinstance(chunk.usage, CompletionUsage)
                        else chunk.usage,
                    )
                    cost = get_abs_cost(usage, ai_model=ai_model)
                    spend += cost
                    yield UsageDetails(
                        id=chunk.id, model=chunk.model, usage=usage, cost=cost
                    )
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        accumulated_content += delta.content
                        yield delta.content
                    reasoning_details = getattr(delta, "reasoning_details", None)
                    if reasoning_details:
                        reasoning_detail_chunks += reasoning_details
                        for obj in reasoning_details:
                            if obj["type"] == "reasoning.summary":
                                yield obj
                            elif obj["type"] == "reasoning.text":
                                yield obj
                    if delta.tool_calls:
                        tool_call_chunks += delta.tool_calls

        message: OpenRouterAssistantMessageParam = {"role": "assistant"}
        if accumulated_content:
            message["content"] = accumulated_content
        if reasoning_detail_chunks:
            message["reasoning_details"] = _gather_reasoning_details_chunks(
                reasoning_detail_chunks
            )
        if tool_call_chunks:
            pending_tool_calls = _gather_tool_call_chunks(tool_call_chunks)
            message["tool_calls"] = cast(
                List[ChatCompletionMessageToolCallUnionParam],
                [tool_call.to_dict() for tool_call in pending_tool_calls],
            )
        else:
            pending_tool_calls = None

        if _is_empty_assistant_message(message):
            logger.error(
                "Model generated an empty assistant message: model=%s", ai_model.name
            )
        messages.append(message)

        if not pending_tool_calls:
            return

        async with aclosing(
            _process_tool_calls(
                messages=messages,
                tools=tools,
                pending_tool_calls=pending_tool_calls,
                spend=spend,
                max_spend=max_spend,
            )
        ) as tool_stream:
            async for chunk in tool_stream:
                yield chunk

        # A response after tool calls gets a fresh history-repair budget.
        async with aclosing(reply()) as next_stream:
            async for chunk in next_stream:
                yield chunk

    # If a consumer stops at yield, aclosing injects GeneratorExit into reply. This
    # exits its OpenRouter stream instead of leaving the generator and HTTP suspended.
    try:
        async with aclosing(reply()) as response_stream:
            async for chunk in response_stream:
                yield chunk
    finally:
        # One client is shared across repaired retries and tool-call responses.
        await client.close()


def _parse_responses_tool_call(
    tools: Optional[List[Callable]],
    tool_call: ResponseFunctionToolCall,
):
    available = {tool.__name__: tool for tool in tools} if tools else {}  # type: ignore
    name = tool_call.name
    if name not in available:
        function = gen_error
        kwargs = {"msg": f"Function name error - unknown name: {name}"}
    else:
        function = available[name]
        try:
            kwargs = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            function = gen_error
            kwargs = {"msg": "Couldn't parse the arguments to the tool"}

    return function, kwargs


def _handle_tool_call_results(
    input_list: ResponseInputParam,
    *,
    tool_call_ids: List[str],
    tool_call_results: List[ToolCallResult | BaseException],
) -> float:
    # Add results to messages
    total_spend = 0
    for tool_call_id, result in zip(tool_call_ids, tool_call_results):
        if isinstance(result, BaseException):
            content = repr(result)
        else:
            if result.get("usage_details"):
                total_spend += result["usage_details"]["cost"]
            content = result["content"]
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": content,
            }
        )
    return total_spend


async def get_response(
    *,
    ai_model: AIModel,
    input: ResponseInputParam,
    tools: List[Callable],
    client: AsyncOpenAI,
    reasoning_effort: Optional[str],
    prompt_cache_key: str | Omit = omit,
    spend: float = 0,
    max_spend: float = 100,
) -> AsyncIterator[EvalEvent | Tuple[Response, float]]:
    response = await client.responses.create(
        model=ai_model.name,
        tools=[_get_response_tools_definition(tool) for tool in tools],
        input=input,
        store=False,
        parallel_tool_calls=True,
        prompt_cache_key=prompt_cache_key,
        **_get_responses_options(get_model_options(ai_model, reasoning_effort)),
    )

    # Extend the existing message history instead of replacing its list.
    input.extend(cast(ResponseInputParam, response.output))
    spend += (
        get_abs_cost(parse_responses_usage(response.usage), ai_model=ai_model)
        if response.usage
        else 0
    )

    # prepare tool calls
    tool_calls = []
    tool_call_ids = []
    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            logger.debug("tool call %s with %s", item.name, item.arguments)
            yield EvalEvent(
                type="dj_evals.event",
                message=f"Tool call: {item.name}\n\n```json\n{item.arguments}\n```",
            )
            function, kwargs = _parse_responses_tool_call(tools, item)
            tool_calls.append((function, kwargs))
            tool_call_ids.append(item.call_id)

    if tool_calls:
        results = []
        async for event in _call_tool_calls_with_events(
            tool_calls, spend=spend, max_spend=max_spend
        ):
            if isinstance(event, list):
                results = event
            else:
                yield event

        spend += _handle_tool_call_results(
            input, tool_call_ids=tool_call_ids, tool_call_results=results
        )

        async for event in get_response(
            ai_model=ai_model,
            input=input,
            tools=tools,
            spend=spend,
            max_spend=max_spend,
            client=client,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
        ):
            yield event
    else:
        logger.debug("total spent using %s = %s", ai_model.name, spend)
        yield response, spend


async def _get_structured_text_response(
    *,
    ai_model: AIModel,
    input: ResponseInputParam,
    tools: List[Callable],
    text_format: type[PydanticModel],
    client: AsyncOpenAI,
    reasoning_effort: Optional[str],
    prompt_cache_key: str | Omit,
    spend: float,
    max_spend: float,
) -> AsyncIterator[EvalEvent | Tuple[ParsedResponse[PydanticModel], float]]:
    compatibility.add_json_schema_to_input(input, text_format)

    async for event in get_response(
        ai_model=ai_model,
        input=input,
        tools=tools,
        spend=spend,
        max_spend=max_spend,
        client=client,
        reasoning_effort=reasoning_effort,
        prompt_cache_key=prompt_cache_key,
    ):
        if not isinstance(event, tuple):
            yield event
            continue

        response, spend = event
        try:
            yield compatibility.parse_response_text(response, text_format), spend
        except ValidationError as error:
            logger.warning(
                "Could not parse structured response from %s: %s. "
                "Retrying for JSON response.",
                ai_model.name,
                error,
            )
            input.append(
                {
                    "role": "user",
                    "content": "Please provide the metrics in the requested JSON format only, without any additional explanation.",
                }
            )

            async for retry_event in get_response(
                ai_model=ai_model,
                input=input,
                tools=tools,
                spend=spend,
                max_spend=max_spend,
                client=client,
                reasoning_effort=reasoning_effort,
                prompt_cache_key=prompt_cache_key,
            ):
                if not isinstance(retry_event, tuple):
                    yield retry_event
                else:
                    response, spend = retry_event
                    yield (
                        compatibility.parse_response_text(response, text_format),
                        spend,
                    )
        return


async def get_structured_response(
    *,
    ai_model: AIModel,
    input: ResponseInputParam,
    tools: List[Callable],
    text_format: type[PydanticModel],
    client: AsyncOpenAI,
    reasoning_effort: Optional[str],
    prompt_cache_key: str | Omit = omit,
    spend: float = 0,
    max_spend: float = 100,
) -> AsyncIterator[EvalEvent | Tuple[ParsedResponse[PydanticModel], float]]:
    if not compatibility.supports_structured_output_with_tools(ai_model):
        async for event in _get_structured_text_response(
            ai_model=ai_model,
            input=input,
            tools=tools,
            text_format=text_format,
            spend=spend,
            max_spend=max_spend,
            client=client,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
        ):
            yield event
        return

    response = await client.responses.parse(
        model=ai_model.name,
        tools=[_get_response_tools_definition(tool) for tool in tools],
        input=input,
        store=False,
        parallel_tool_calls=True,
        text_format=text_format,
        prompt_cache_key=prompt_cache_key,
        **_get_responses_options(get_model_options(ai_model, reasoning_effort)),
    )

    # Extend the existing message history instead of replacing its list.
    input.extend(cast(ResponseInputParam, response.output))
    spend += (
        get_abs_cost(parse_responses_usage(response.usage), ai_model=ai_model)
        if response.usage
        else 0
    )

    # prepare tool calls
    tool_calls = []
    tool_call_ids = []
    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            logger.debug("tool call %s with %s", item.name, item.arguments)
            yield EvalEvent(
                type="dj_evals.event",
                message=f"Tool call: {item.name}\n\n```json\n{item.arguments}\n```",
            )
            function, kwargs = _parse_responses_tool_call(tools, item)
            tool_calls.append((function, kwargs))
            tool_call_ids.append(item.call_id)

    if tool_calls:
        results = []
        async for event in _call_tool_calls_with_events(
            tool_calls, spend=spend, max_spend=max_spend
        ):
            if isinstance(event, list):
                results = event
            else:
                yield event

        spend += _handle_tool_call_results(
            input, tool_call_ids=tool_call_ids, tool_call_results=results
        )

        async for event in get_structured_response(
            ai_model=ai_model,
            input=input,
            tools=tools,
            spend=spend,
            max_spend=max_spend,
            client=client,
            reasoning_effort=reasoning_effort,
            text_format=text_format,
            prompt_cache_key=prompt_cache_key,
        ):
            yield event
    else:
        logger.debug("total spent using %s = %s", ai_model.name, spend)
        yield response, spend
