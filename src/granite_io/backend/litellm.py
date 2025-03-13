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

    def process_input(self, **kwargs: Any) -> Dict[str, Any]:
        # TODO prompt: is required.  Want args/kwargs? (model, prompt, **kwargs)

        # Add required model, if missing
        if not kwargs.get("model"):
            kwargs["model"] = self._model_str

        # From ChatCompletionInputs we have extra stuff that is not model input
        kwargs.pop("messages", None)
        kwargs.pop("tools", None)
        kwargs.pop("thinking", None)

        # Migrate alias kwargs to this flavor of backend
        self.kwarg_alias(kwargs, "stop", "stop_strings")
        self.kwarg_alias(kwargs, "n", "num_return_sequences")

        #
        # Note: Questionable validity checking -- this could be left up to the model
        #

        # LiteLLM throws an error if stop is str and not array
        if isinstance(kwargs.get("stop"), str):
            kwargs["stop"] = [kwargs["stop"]]

        # n (a.k.a. num_return_sequences) validation (n >= 1)
        # Setting n requires setting best_of >= n
        n = kwargs.get("n")  # Allow default for missing/None
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

        with import_optional("litellm"):
            # Third Party
            import litellm

        return await litellm.atext_completion(**kwargs)

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
