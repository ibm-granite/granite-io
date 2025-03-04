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

    async def generate(self, **kwargs) -> GenerateResults:
        """Run a direct /completions call"""

        # model is required
        if not kwargs.get("model"):
            kwargs["model"] = self._model_str

        # Normalize num_return_sequences a.k.a "n"
        n = kwargs.get("n")
        num_return_sequences = kwargs.get("num_return_sequences")

        # "n" gets priority because that is the OpenAI kwarg
        # "num_return_sequences" needs to be removed/renamed
        if num_return_sequences is not None:
            del kwargs["num_return_sequences"]
            if not n:
                kwargs["n"] = num_return_sequences

        #
        # Questionable validity checking -- this could be left up to the model
        #

        # TODO:
        # frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        # presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        # stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        # temperature: Optional[float] | NotGiven = NOT_GIVEN,

        # n (a.k.a. num_return_sequences) validation
        n = kwargs.get("n")

        if n is not None:
            if not isinstance(n, int) or n < 1:  # Check like the others for invalid
                raise ValueError(f"Invalid value for n ({n})")
            elif n > 1:
                # best_of must be >= n
                best_of = kwargs.get("best_of")
                if not isinstance(best_of, int) or best_of < n:
                    kwargs["best_of"] = n

        result = await self.openai_client.completions.create(**kwargs)

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
