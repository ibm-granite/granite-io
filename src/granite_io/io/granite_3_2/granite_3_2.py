# SPDX-License-Identifier: Apache-2.0

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.io.base import ModelDirectInputOutputProcessor
from granite_io.io.consts import (
    _GRANITE_3_2_2B_HF,
    _GRANITE_3_2_2B_OLLAMA,
    _GRANITE_3_2_MODEL_NAME,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2InputProcessor,
)
from granite_io.io.granite_3_2.output_processors.granite_3_2_output_processor import (
    Granite3Point2OutputProcessor,
)
from granite_io.io.registry import io_processor
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResults,
    GenerateResults,
)


@io_processor(
    _GRANITE_3_2_MODEL_NAME,
    # Huggingface
    _GRANITE_3_2_2B_HF,
    "ibm-granite/granite-3.2-8b-instruct",
    # Ollama
    "granite3.2",
    "granite3.2:8b",
    _GRANITE_3_2_2B_OLLAMA,
    # RITS
    "ibm-granite/granite-8b-instruct-preview-4k",
)
class Granite3Point2InputOutputProcessor(ModelDirectInputOutputProcessor):
    """
    Input/output processor for version 3.2 of the main Granite models, all sizes.

    This input/output processor is based on the Jinja template that was used during
    supervised fine tuning of these models.
    """

    def __init__(self, config: aconfig.Config = None, backend: Backend | None = None):
        """
        :param backend: Handle on inference engine, required if this io processor's
            :func:`create_chat_completion()` method is going to be used
        """
        super().__init__(backend=backend)

    def inputs_to_string(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        input_processor = Granite3Point2InputProcessor()
        return input_processor.transform(inputs, add_generation_prompt)

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        output_processor = Granite3Point2OutputProcessor()
        return output_processor.transform(output, inputs)
