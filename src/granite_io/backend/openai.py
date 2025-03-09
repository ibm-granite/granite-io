# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.optional import import_optional
from granite_io.types import GenerateResult, GenerateResults

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

    async def generate(
        self, input_str: str, num_return_sequences: int = 1
    ) -> GenerateResults:
        """Run a direct /completions call"""

        if num_return_sequences < 1:
            raise ValueError(
                f"Invalid value for num_return_sequences ({num_return_sequences})"
            )

        result = await self._openai_client.completions.create(
            model=self._model_str,
            prompt=input_str,
            best_of=num_return_sequences,
            n=num_return_sequences,
            max_tokens=1024,  # TODO: make this configurable
        )
        results = []
        for choice in result.choices:
            results.append(
                GenerateResult(
                    completion_string=choice.text,
                    completion_tokens=[],  # Not part of the OpenAI spec
                    stop_reason=choice.finish_reason,
                )
            )

        return GenerateResults(results=results)
