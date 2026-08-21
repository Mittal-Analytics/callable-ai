import json
from typing import TypeVar

from openai.types.responses import ParsedResponse, Response, ResponseInputParam
from openai.types.responses.parsed_response import (
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
)
from pydantic import BaseModel

from maai.models import AIModel

# Keeps the parsed response type tied to the Pydantic schema passed in.
PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


def supports_structured_output_with_tools(ai_model: AIModel) -> bool:
    name = ai_model.name.lower()
    if any(model in name for model in ["deepseek", "minimax"]):
        return False
    if "/" not in name:
        return True
    return any(model in name for model in ["gemini", "gpt", "grok"])


def add_json_schema_to_input(input: ResponseInputParam, text_format: type[BaseModel]):
    schema = json.dumps(text_format.model_json_schema(), indent=2)
    input.append(
        {
            # User role is widely supported by OpenAI-compatible providers.
            "role": "user",
            "content": "Return only valid JSON matching this schema:"
            f"\n```json\n{schema}\n```",
        }
    )


def parse_response_text(
    response: Response, text_format: type[PydanticModel]
) -> ParsedResponse[PydanticModel]:
    """Build ParsedResponse for providers that return JSON as plain text."""
    parsed = text_format.model_validate_json(get_json_text(response.output_text))

    # ParsedResponse.output_parsed reads this `parsed` field internally.
    parsed_text = ParsedResponseOutputText[PydanticModel](
        annotations=[],
        text=response.output_text,
        type="output_text",
        parsed=parsed,
    )

    # Keep the same shape as OpenAI's parsed message response.
    parsed_message = ParsedResponseOutputMessage[PydanticModel](
        id=response.id,
        content=[parsed_text],
        role="assistant",
        status="completed",
        type="message",
    )

    # Reuse all original response fields but replace output with parsed output.
    data = response.model_dump()
    data["output"] = [parsed_message]
    return ParsedResponse[PydanticModel].model_construct(**data)


def get_json_text(text: str) -> str:
    text = text.strip()
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                return block[4:].strip()
            if block.startswith(("{", "[")):
                return block

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end >= object_start:
        return text[object_start : object_end + 1]

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end >= array_start:
        return text[array_start : array_end + 1]
    return text
