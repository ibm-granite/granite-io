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

    async def generate(self, **kwargs) -> GenerateResults:
        """Run a direct /completions call"""

        # model is required
        if not kwargs.get("model"):
            kwargs["model"] = self._model_str

        # Normalize num_return_sequences a.k.a "n"
        n = kwargs.get("n")
        num_return_sequences = kwargs.get("num_return_sequences")

        # "n" gets priority because that is the LiteLLM kwarg
        # "num_return_sequences" needs to be removed/renamed
        if num_return_sequences is not None:
            del kwargs["num_return_sequences"]
            if not n:
                kwargs["n"] = num_return_sequences

        #
        # Questionable validity checking -- this could be left up to the model
        #

        # TODO prompt: is required

        # "Optional" model strings start with provider/
        #    model="watsonx/ibm/granite-3-2-8b-instruct"
        #    model="ollama/llama3.1:latest"

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

        # temperature: Optional[float] = None,  # Optional: Sampling temperature to use.
        # presence_penalty: Optional[  -- is this repetition_penalty?
            # float
        # ] = None,  # Optional: Penalize new tokens based on whether they appear in the text so far.
        # stop strings
        # stop: Optional[
            # Union[str, List[str]]
        # ] = None,

        with import_optional("litellm"):
            # Third Party
            import litellm

        result = await litellm.text_completion(**kwargs)

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
