# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.optional import import_optional
from granite_io.types import ChatCompletionInputs, GenerateResult

if TYPE_CHECKING:
    # Third Party
    import openai


@backend(
    "openai",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
            "openai_base_url": {"type": "string"},
            "openai_api_key": {"type": "string"},
        }
    },
)
class OpenAIBackend(Backend):
    _model_str: str
    _openai_client: "openai.AsyncOpenAI"

    def __init__(self, config: aconfig.Config):
        with import_optional("openai"):
            # Third Party
            import openai

        self._model_str = config.model_name
        api_key = config.get("openai_api_key", "ollama")
        base_url = config.get("openai_base_url", "http://localhost:11434/v1")
        default_headers = {"RITS_API_KEY": api_key} if api_key else None

        self._openai_client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, default_headers=default_headers
        )

    async def create_chat_completion(self, input_chat: ChatCompletionInputs) -> str:
        messages = [{"role": x.role, "content": x.content} for x in input_chat.messages]

        result = await self._openai_client.chat.completions.create(
            model=self._model_str,
            messages=messages,
            store=True,
        )

        raw_message = result.choices[0].message

        if raw_message.role != "assistant":
            raise ValueError(f"Unexpected role '{raw_message.role}' in chat completion")
        if raw_message.content is None:
            raise NotImplementedError(
                f"No text in chat completion, and decoding of other types is not "
                f"implemented. Completion result: {raw_message}"
            )

        # # return before_state.append(AssistantMessage(raw_message.content))
        # TODO: don't we want this stuff too?
        # return GenerateResult(
        # completion_string=raw_message.content,
        # completion_tokens=result.usage.completion_tokens,
        # stop_reason=result.choices[0].finish_reason
        # )

        return raw_message.content

    async def generate(self, input_str: str) -> GenerateResult:
        result = await self._openai_client.completions.create(
            model=self._model_str,
            prompt=input_str,
        )
        return GenerateResult(
            completion_string=result.choices[0].text,
            completion_tokens=[],  # Not part of the OpenAI spec
            stop_reason=result.choices[0].finish_reason,
        )
