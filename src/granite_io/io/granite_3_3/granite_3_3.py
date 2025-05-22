# SPDX-License-Identifier: Apache-2.0

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.io.base import ModelDirectInputOutputProcessor
from granite_io.io.consts import (
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_2B_OLLAMA,
    _GRANITE_3_3_8B_HF,
    _GRANITE_3_3_8B_OLLAMA,
    _GRANITE_3_3_MODEL_NAME,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3InputProcessor,
)
from granite_io.io.granite_3_3.output_processors.granite_3_3_output_processor import (
    Granite3Point3OutputProcessor,
)
from granite_io.io.registry import io_processor
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)


@io_processor(
    _GRANITE_3_3_MODEL_NAME,
    # Huggingface
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_8B_HF,
    # Ollama
    "granite3.3",
    _GRANITE_3_3_2B_OLLAMA,
    _GRANITE_3_3_8B_OLLAMA,
)
class Granite3Point3InputOutputProcessor(ModelDirectInputOutputProcessor):
    """
    Input/output processor for version 3.3 of the main Granite models, all sizes.

    This input/output processor is based on the Jinja template that was used during
    supervised fine-tuning of these models.
    """

    def __init__(self, config: aconfig.Config = None, backend: Backend | None = None):
        """
        :param backend: Handle on inference engine, required if this io processor's
            :func:`create_chat_completion()` method is going to be used
        """
        super().__init__(backend=backend)

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

        # Add skip_special_tokens = False to see citations.
        if inputs.controls and inputs.controls.citations is not None:
            extra_body = generate_inputs.extra_body or {}
            extra_body["skip_special_tokens"] = False
            generate_inputs.extra_body = extra_body
        model_output = await self._backend.pipeline(generate_inputs)
        return self.output_to_result(output=model_output, inputs=inputs)

    def inputs_to_string(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        input_processor = Granite3Point3InputProcessor()
        return input_processor.transform(inputs, add_generation_prompt)

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        output_processor = Granite3Point3OutputProcessor()
        return output_processor.transform(output, inputs)
