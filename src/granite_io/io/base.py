# SPDX-License-Identifier: Apache-2.0

"""
Base classes for Input/Output processors
"""

# Standard
import abc
import asyncio
import threading

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend, ChatCompletionBackend
from granite_io.factory import FactoryConstructible
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)


def _workaround_for_horrible_design_flaw_in_asyncio(coroutine_to_run):
    """
    Horrible hack that seems to be the only way to call a coroutine from a non-async
    function that is being called from an async function.

    Creates a background thread with its own event loop. The background thread then
    runs a single task in that event loop.
    """
    result_holder = []

    def _callback():
        # Python's thread library requires you to write your own code to return results
        # or raise exceptions from a background thread.
        try:
            result_holder.append(asyncio.run(coroutine_to_run))
        except Exception as e:  # pylint: disable=W0718
            result_holder.append(e)

    thread = threading.Thread(target=_callback)
    thread.start()
    thread.join()
    if isinstance(result_holder[0], Exception):
        raise result_holder[0]
    return result_holder[0]


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
    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        """
        :param inputs: Structured representation of the inputs to a chat completion
            request, possibly including additional fields that only this input-output
            processor can consume

        :returns: The next message that the model produces when fed the specified
            inputs, plus additional information about the low-level request.
        """

    def create_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        """
        Non-async version of :func:`acreate_chat_completion()`

        :param inputs: Structured representation of the inputs to a chat completion
            request, possibly including additional fields that only this input-output
            processor can consume

        :returns: The next message that the model produces when fed the specified
            inputs, plus additional information about the low-level request.
        """
        # Fall back on async version of this method by default.  Subclasses may override
        # this method if they have a more efficient way of doing non-async operation.
        coroutine_to_run = self.acreate_chat_completion(inputs)
        try:  # Exceptions as control flow. Sorry, asyncio forces this design on us.
            asyncio.get_running_loop()

            # If we get here, this code is running inside an async function.
            return _workaround_for_horrible_design_flaw_in_asyncio(coroutine_to_run)
        except RuntimeError:
            # If we get here, this code is not running inside an async function.
            # First we exit the the exception handler; otherwise any exceptions that are
            # thrown from the coroutine will be chained off the current RuntimeError.
            pass
        return asyncio.run(coroutine_to_run)


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

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        if self._backend is None:
            raise ValueError(
                "Attempted to call create_chat_completion() without "
                "configuring an inference backend."
            )

        # Do not modify `inputs` in place. It is VERY VERY IMPORTANT not to do so.
        # The caller may be using that object for something else.
        generate_inputs = (
            inputs.generate_inputs.model_copy()
            if inputs.generate_inputs is not None
            else GenerateInputs()
        )
        generate_inputs.prompt = self.inputs_to_string(inputs)
        model_output = await self._backend.pipeline(generate_inputs)
        return self.output_to_result(output=model_output, inputs=inputs)

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
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
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


class ModelDirectInputOutputProcessorWithGenerate(InputOutputProcessor):
    """
    Base class for IO processors that directly invoke a model and that modify the
    generation arguments of requests, such as by enabling constrained decoding.
    """

    _backend: Backend | None
    """Handle on the current inference engine, required if this io processor's
    :func:`acreate_chat_completion()` method is going to be used"""

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

    @abc.abstractmethod
    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        """
        Determine the best generation parameters (including prompt) to pass to the
        backend when running the specified chat completion.

        :param inputs: Structured representation of the inputs
        :param add_generation_prompt: If true, the generated prompt will include the
         appropriate response prefix to trigger generation.

        :returns: A copy of inputs.generate_inputs with appropriate modifications for
         the target model and with the ``prompt`` field populated with an appropriate
         prompt
        """

    @abc.abstractmethod
    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
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

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        if self._backend is None:
            raise ValueError(
                "Attempted to call acreate_chat_completion() without "
                "configuring an inference backend."
            )

        # Copy the inputs to prevent the caller experiencing surprise modifications to a
        # local variable.
        inputs = inputs.model_copy()
        generate_inputs = self.inputs_to_generate_inputs(inputs)
        model_output = await self._backend.pipeline(generate_inputs)
        return self.output_to_result(output=model_output, inputs=inputs)


class InputProcessor(FactoryConstructible):
    """
    Interface for generic input processors. An input processor exposes an
    API to transform model completion request into a string representation.
    """

    def __init__(self, config: aconfig.Config | None = None):  # pylint: disable=unused-argument
        """By default an input processor doesn't require config"""

    @abc.abstractmethod
    def transform(
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


class OutputProcessor(FactoryConstructible):
    """
    Interface for generic output processors. An putput processor exposes an
    API to transform model output into a structured representation of the
    information.
    """

    def __init__(self, config: aconfig.Config | None = None):  # pylint: disable=unused-argument
        """By default an output processor doesn't require config"""

    @abc.abstractmethod
    def transform(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        """
        Convert the model output generated into a structured representation of the
        information.

        :param output: Output of the a generation request, potentially incomplete
           if it was a streaming request
        :param inputs: Optional reference to the inputs that caused the model to produce
           ``output``, for validating the correctness of the output. If no inputs are
           provided, this method may skip some validations but will still produce the
           same result.

        :returns: The parsed output so far
        """


def make_new_io_processor(
    input_processor: InputProcessor,
    output_processor: OutputProcessor,
    config: aconfig.Config = None,
    backend: Backend | None = None,
) -> ModelDirectInputOutputProcessor:
    """
    Wrapper function that creates an instance of an InputOutputProcessor based on the
    InputProcessor and OutputProcessor passed in the function call.

    :param input_processor: Processor that performs processing of input to model
    :param input_processor: Processor that performs processing of output from the model
    :param config: Setup config for this IO processor
    :param backend: Handle on inference engine, required if this io processor's
        :func:`create_chat_completion()` method is going to be used

    :returns: The IO processor

    :raise: ValueError - If input or output processor is None
    """

    class _InputOutputProcessor(ModelDirectInputOutputProcessor):
        """
        InputOutputProcessor template.

        This InputOutputProcessor is based on the input and the output processors
        passed during creation.
        """

        def __init__(
            self,
            input_processor: InputProcessor,
            output_processor: OutputProcessor,
            config: aconfig.Config = None,
            backend: Backend | None = None,
        ):
            """
            :param input_processor: Processor that performs processing of input to
                model
            :param input_processor: Processor that performs processing of output from
                the model
            :param config: Setup config for this IO processor
            :param backend: Handle on inference engine, required if this io processor's
                :func:`create_chat_completion()` method is going to be used

            :raise: ValueError - If input or output processor is None
            """
            super().__init__(config=config, backend=backend)

            self._input_processor = input_processor
            self._output_processor = output_processor

            if self._input_processor is None:
                raise ValueError(
                    "Attempted to create IO Processor without "
                    "setting an Input Processor ."
                )
            if self._output_processor is None:
                raise ValueError(
                    "Attempted to create IO Processor without "
                    "setting an Output Processor ."
                )

        def inputs_to_string(
            self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
        ) -> str:
            return self._input_processor.transform(inputs, add_generation_prompt)

        def output_to_result(
            self,
            output: GenerateResults,
            inputs: ChatCompletionInputs | None = None,
        ) -> ChatCompletionResults:
            return self._output_processor.transform(output, inputs)

    return _InputOutputProcessor(
        input_processor=input_processor,
        output_processor=output_processor,
        config=config,
        backend=backend,
    )


class RequestProcessor(abc.ABC):
    """
    Base class for objects that modify a ``ChatCompletionInputs`` object in various
    ways, such as by adding RAG documents or changing the content of existing turns.
    """

    @abc.abstractmethod
    async def aprocess(
        self, inputs: ChatCompletionInputs
    ) -> list[ChatCompletionInputs]:
        """
        Subclasses must implement this entry point. Modify the request, potentially
        spawning asynchronous background tasks to perform the modifications.

        :param inputs: The original request

        :returns: One or more modified COPIES of the original request
        """

    def process(self, inputs: ChatCompletionInputs) -> list[ChatCompletionInputs]:
        """
        Subclasses may optionally implement this entry point if they have a more
        efficient way to do non-async operation. The default implementation calls to
        :func:`aprocess()`.

        :param inputs: The original request

        :returns: One or more modified COPIES of the original request
        """
        # Fall back on async version of this method by default.  Subclasses may override
        # this method if they have a more efficient way of doing non-async operation.
        coroutine_to_run = self.aprocess(inputs)
        try:  # Exceptions as control flow. Sorry, asyncio forces this design on us.
            asyncio.get_running_loop()

            # If we get here, this code is running inside an async function.
            return _workaround_for_horrible_design_flaw_in_asyncio(coroutine_to_run)
        except RuntimeError:
            # If we get here, this code is not running inside an async function.
            # First we exit the the exception handler; otherwise any exceptions that are
            # thrown from the coroutine will be chained off the current RuntimeError.
            pass
        return asyncio.run(coroutine_to_run)
