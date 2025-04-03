# SPDX-License-Identifier: Apache-2.0

"""
Base class for backends
"""

# Standard
import abc

# Third Party
import aconfig

# Local
from granite_io.factory import FactoryConstructible
from granite_io.types import (
    ChatCompletionInputs,
    GenerateInputs,
    GenerateResult,
    GenerateResults,
)


# We put this base class here to avoid circular imports
class Backend(FactoryConstructible):
    """
    Base class for classes that provide an interface to a string-based completions API
    for a model.

    This base class exists for two reasons: It smoothes out the differences between
    APIs, and it provides a level of indirection so that backend inference libraries can
    be optional dependencies of this Python package.
    """

    _model_str: str

    def __init__(self, config: aconfig.Config):
        self._model_str = config.model_name

    async def __call__(self, inputs: GenerateInputs) -> GenerateResults:
        return await self.pipeline(inputs)

    async def pipeline(self, inputs: GenerateInputs) -> GenerateResults:
        """
        Process input, call completion (generate), process and return output
        """
        inputs = self.process_input(inputs)
        output = await self.generate(inputs)
        return self.process_output(output)

    def process_input(self, inputs: GenerateInputs) -> GenerateInputs:
        """
        Process inputs to prepare them for completion.create() (or generate())

        Args:
            inputs (GenerateInputs): the inputs which are being processed
             to be used in target backend call.

        Returns:
            GenerateInputs: a new copy of inputs, modified for the Backend
        """

        # Do not modify `inputs` in place.
        inputs_copy = inputs.model_copy() if inputs is not None else GenerateInputs()

        # Add required model, if missing
        if not inputs_copy.model:
            inputs_copy.model = self._model_str

        # n (a.k.a. num_return_sequences) validation
        n = inputs_copy.n
        if n is not None:  # noqa SIM102
            if n < 1:
                raise ValueError(f"Invalid value for n ({n})")
            if n > 1:
                # best_of must be >= n
                best_of = inputs_copy.best_of
                if best_of is None or best_of < n:
                    inputs_copy.best_of = n

        # Some backends prefer an array to a string
        if isinstance(inputs_copy.stop, str):
            inputs_copy.stop = [inputs_copy.stop]

        return inputs_copy

    @abc.abstractmethod
    async def generate(self, inputs: GenerateInputs) -> GenerateResults:
        """
        Callback to invoke the model to generate a response.
        """
        raise NotImplementedError()

    def process_output(self, outputs):
        """
        Process output from completion.create() (or generate())
        """
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

    @staticmethod
    def kwarg_alias(kw_dict, preferred_key, alias_key):
        """Migrate kwarg k,v from alias key to preferred key"""
        if alias_key in kw_dict:
            alias_value = kw_dict.get(alias_key)
            # Alias is removed
            del kw_dict[alias_key]
            # Preferred key/value gets priority if both are specified
            # TODO: optionally error on redundant kwargs(?)
            if preferred_key not in kw_dict:
                # Alias value is set to preferred key (iff not already set)
                kw_dict[preferred_key] = alias_value
        return kw_dict  # return reference for convenience


class ChatCompletionBackend(FactoryConstructible):
    """
    Base class for classes that provide an interface to a chat completions API.

    This base class exists for two reasons: It smooths out the differences between
    APIs, and it provides a level of indirection so that backend inference libraries can
    be optional dependencies of this Python package.
    """

    @abc.abstractmethod
    async def create_chat_completion(self, input_chat: ChatCompletionInputs) -> str:
        """
        Callback to invoke the model to generate a response.
        """
        raise NotImplementedError()
