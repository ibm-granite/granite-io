# SPDX-License-Identifier: Apache-2.0

# Third Party
from openai import AsyncOpenAI
import aconfig

# Local
from granite_io.backend.base import Backend, ChatCompletionBackend
from granite_io.backend.registry import backend
from granite_io.types import ChatCompletionInputs, GenerateResult


@backend(
    "openai",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
        }
    },
)
class OpenAIBackend(Backend, ChatCompletionBackend):
    _model_str: str
    _openai_client: AsyncOpenAI

    def __init__(self, config: aconfig.Config):
        self._model_str = config.model_name

        # TODO: cleanup this W-I-P
        # Use openai env vars, but use local ollama defaults instead of openai
        # Standard
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        # TODO: cleanup this W-I-P

    async def create_chat_completion(self, input_chat: ChatCompletionInputs) -> str:
        messages = [{"role": x.role, "content": x.content} for x in input_chat.messages]

        result = await self.openai_client.chat.completions.create(
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
        """Run a direct /completions call"""

        result = await self.openai_client.completions.create(
            model=self._model_str,
            prompt=input_str,
        )
        return GenerateResult(
            completion_string=result.choices[0].text,
            completion_tokens=[],  # Not part of the OpenAI spec
            stop_reason=result.choices[0].finish_reason,
        )
