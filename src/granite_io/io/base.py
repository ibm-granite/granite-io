# SPDX-License-Identifier: Apache-2.0

"""
Base classes for Input/Output processors
"""

# Standard
import abc

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend, ChatCompletionBackend
from granite_io.factory import FactoryConstructible
from granite_io.types import ChatCompletionInputs, ChatCompletionResult


class InputOutputProcessor(FactoryConstructible):
    """
    Interface for generic input-output processors. An input-output processor exposes an
    API similar to a chat completions API. Each "completion" call may result in multiple
    LLM calls or even zero LLM calls, depending on the implementation of the
    input-output processor.
    """

    def __init__(self, config: aconfig.Config | None = None):  # pylint: disable=unused-argument
        """By default an IO processor doesn't require config"""

    @abc.abstractmethod
    async def create_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResult:
        """
        :param inputs: Structured representation of the inputs to a chat completion
            request, possibly including additional fields that only this input-output
            processor can consume

        :returns: The next message that the model produces when fed the specified
            inputs, plus additional information about the low-level request.
        """


class ModelDirectInputOutputProcessor(InputOutputProcessor):
    """
    Base class for input-output processors that work by translating inputs to strings
    suitable for tokenization, then passing these strings to a particular model's
    low-level completions API, then parsing raw output strings.
    """

    _backend: Backend | ChatCompletionBackend | None
    """Handle on the current inference engine, required if this io processor's
    :func:`create_chat_completion()` method is going to be used"""

    def __init__(
        self, config: aconfig.Config | None = None, backend: Backend | None = None
    ):
        """
        :param config: Setup config for this IO processor
        :param backend: Handle on inference engine, required if this io processor's
            :func:`create_chat_completion()` method is going to be used
        """
        super().__init__(config)
        self._backend = backend

    async def create_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResult:
        if self._backend is None:
            raise ValueError(
                "Attempted to call create_chat_completion() without "
                "configuring an inference backend."
            )
        input_string = self.inputs_to_string(inputs)
        generation_output = await self._backend.generate(input_string)
        return self.output_to_result(generation_output.completion_string, inputs)

    @abc.abstractmethod
    def inputs_to_string(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        """
        Convert the structured representation of the inputs to a completion request into
        the string representation of the tokens that should be sent to the model to
        implement said request.

        :param inputs: Structured representation of the inputs
        :param add_generation_prompt: If ``True``, the returned prompt string will
            contain a prefix of the next assistant response for use as a prompt to a
            generation request. Otherwise, the prompt will only contain the messages and
            documents in ``input``.

        :returns: String that can be passed to the model's tokenizer to create a prompt
            for generation.
        """

    @abc.abstractmethod
    def output_to_result(
        self, output: str, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResult:
        """
        Convert the structured representation of the inputs to a completion request into
        the string representation of the tokens that should be sent to the model to
        implement said request.

        :param output: Output of the a generation request, potentially incomplete
           if it was a streaming request
        :param inputs: Optional reference to the inputs that caused the model to produce
           ``output``, for validating the correctness of the output. If no inputs are
           provided, this method may skip some validations but will still produce the
           same result.

        :returns: The parsed output so far
        """
