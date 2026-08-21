from openai.types.responses.response_usage import ResponseUsage

from mai.models import AIModel
from mai.openrouter import OpenRouterCompletionUsage


def parse_responses_usage(responses_usage: ResponseUsage) -> OpenRouterCompletionUsage:
    """
    Convert OpenAI Responses API usage format to OpenRouter completion usage format.
    """
    openrouter_usage: OpenRouterCompletionUsage = {
        "prompt_tokens": responses_usage.input_tokens,
        "prompt_tokens_details": {
            "cached_tokens": responses_usage.input_tokens_details.cached_tokens
        },
        "completion_tokens": responses_usage.output_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": responses_usage.output_tokens_details.reasoning_tokens
        },
        "total_tokens": responses_usage.total_tokens,
    }
    return openrouter_usage


def get_abs_cost(
    usage: OpenRouterCompletionUsage,
    ai_model: AIModel,
    *,
    number_of_web_searches=0,
) -> float:
    # https://platform.openai.com/docs/api-reference/chat-streaming/streaming#chat-streaming/streaming-usage
    input_tokens = usage["prompt_tokens"]
    prompt_details = usage.get("prompt_tokens_details") or {}
    input_tokens_cached = prompt_details.get("cached_tokens") or 0
    output_tokens = usage["completion_tokens"]
    completion_details = usage.get("completion_tokens_details") or {}
    output_tokens_reasoning = completion_details.get("reasoning_tokens") or 0

    input_fresh_tokens = input_tokens - input_tokens_cached
    output_answer_tokens = output_tokens - output_tokens_reasoning

    if ai_model.lower_token_count_cost:
        cutoff, lower_token_ai_model = ai_model.lower_token_count_cost
        if input_tokens <= cutoff:
            ai_model = lower_token_ai_model

    if number_of_web_searches:
        web_search_cost = number_of_web_searches * ai_model.web_search_cost_inr / 1000
    else:
        web_search_cost = 0

    # Calculate cost in INR/per million tokens
    cost = (
        (
            input_fresh_tokens * ai_model.input_tokens_cost_inr
            + output_answer_tokens * ai_model.output_tokens_cost_inr
            + input_tokens_cached * ai_model.input_tokens_cached_cost_inr
            + output_tokens_reasoning * ai_model.output_tokens_reasoning_cost_inr
        )
        / 1_000_000
    ) + web_search_cost
    return abs(cost)
