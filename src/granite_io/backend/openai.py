# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any, Dict

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

    def process_input(self, **kwargs: Any) -> Dict[str, Any]:
        # From ChatCompletionInputs we have extra stuff that is not model input
        kwargs.pop("messages", None)
        kwargs.pop("tools", None)
        kwargs.pop("thinking", None)

        # model is required
        if not kwargs.get("model"):
            kwargs["model"] = self._model_str

        # Migrate alias kwargs to this flavor of backend
        self.kwarg_alias(kwargs, "stop", "stop_strings")
        self.kwarg_alias(kwargs, "n", "num_return_sequences")

        #
        # Questionable validity checking -- this could be left up to the model
        #

        # n (a.k.a. num_return_sequences) validation
        n = kwargs.get("n")

        if n is not None:  # noqa SIM102
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"Invalid value for n ({n})")
            if n > 1:
                # best_of must be >= n
                best_of = kwargs.get("best_of")
                if not isinstance(best_of, int) or best_of < n:
                    kwargs["best_of"] = n

        return kwargs

    async def generate(self, **kwargs):
        """Run a direct /completions call"""
        # pylint: disable-next=missing-kwoa
        return await self._openai_client.completions.create(**kwargs)

    def process_output(self, output, **kwargs):
        results = []
        for choice in output.choices:
            results.append(
                GenerateResult(
                    completion_string=choice.text,
                    completion_tokens=[],  # Not part of the OpenAI spec
                    stop_reason=choice.finish_reason,
                )
            )

        return GenerateResults(results=results)
