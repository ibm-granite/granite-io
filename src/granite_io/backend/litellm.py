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
    import litellm  # noqa: F401


@backend(
    "litellm",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
            # "openai_base_url": {"type": "string"},
            # "openai_api_key": {"type": "string"},
        }
    },
)
class LiteLLMBackend(Backend):
    _model_str: str

    def __init__(self, config: aconfig.Config):
        self._model_str = config.model_name

    def generate(
        self, input_str: str, num_return_sequences: int = 1
    ) -> GenerateResults:
        """Run a direct /completions call"""

        if num_return_sequences < 1:  # Check like the others for invalid
            raise ValueError(
                f"Invalid value for num_return_sequences ({num_return_sequences})"
            )

        with import_optional("litellm"):
            # Third Party
            import litellm

        result = litellm.text_completion(
            # model strings start with provider/
            # model="watsonx/ibm/granite-3-2-8b-instruct",
            # model="ollama/llama3.1:latest",
            model=self._model_str,
            prompt=input_str,
            best_of=num_return_sequences,
            n=num_return_sequences,
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
