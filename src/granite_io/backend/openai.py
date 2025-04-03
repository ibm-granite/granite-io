# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.optional import import_optional
from granite_io.types import GenerateInputs, GenerateResult, GenerateResults

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
    _openai_client: "openai.AsyncOpenAI"

    def __init__(self, config: aconfig.Config):
        super().__init__(config)
        with import_optional("openai"):
            # Third Party
            import openai

        api_key = config.get("openai_api_key", "ollama")
        base_url = config.get("openai_base_url", "http://localhost:11434/v1")
        default_headers = config.get("default_headers")

        self._openai_client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, default_headers=default_headers
        )

    async def generate(self, inputs: GenerateInputs):
        """Run a direct /completions call"""

        # Implementations of the OpenAI APIs tend to get upset if you pass them null
        # values, particularly if those values are associated with arguments that the
        # implementation doesn't support. So strip out null parameters.
        args_dict = inputs.model_dump(exclude_none=True)
        # pylint: disable-next=missing-kwoa
        return await self._openai_client.completions.create(**args_dict)

    def process_output(self, outputs):
        results = []
        for choice in outputs.choices:
            results.append(
                GenerateResult(
                    completion_string=choice.text,
                    completion_tokens=[],  # Not part of the OpenAI spec
                    stop_reason=choice.finish_reason,
                )
            )

        return GenerateResults(results=results)
