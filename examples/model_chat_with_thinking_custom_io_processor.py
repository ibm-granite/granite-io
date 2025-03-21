# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model. It uses the framework capability
to create custom IO processor using a specific input processor and output
processor.

In this scenario the chat request enables thinking mode in the model
to provide better understanding of how the model arrived at its answer.
"""

# Third Party
import aconfig

# Local
from granite_io import (  # make_new_io_processor,
    Backend,
    ChatCompletionInputs,
    ChatCompletionResults,
    GenerateResults,
    ModelDirectInputOutputProcessor,
    UserMessage,
    get_input_processor,
    get_output_processor,
    make_backend,
)

ollama_model_name = "granite3.2:8b"

# Create IO processor using specified input and output processors
# io_processor = make_new_io_processor(
#    input_processor=get_input_processor(ollama_model_name),
#    output_processor=get_output_processor(ollama_model_name),
#    backend=make_backend("openai", {"model_name": ollama_model_name}),
# )


class _MyInputOutputProcessor(ModelDirectInputOutputProcessor):
    """
    Custom IO processor which uses specifc Granite 3.2 input and output
    processors.
    """

    def __init__(
        self,
        model_name: str,
        config: aconfig.Config = None,
        backend: Backend | None = None,
    ):
        """
        :param backend: Handle on inference engine, required if this io processor's
            :func:`create_chat_completion()` method is going to be used
        :param config: Setup config for this IO processor
        """
        super().__init__(backend=backend)
        self._model_name = model_name

    def inputs_to_string(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        input_processor = get_input_processor(self._model_name)
        return input_processor.transform(inputs, add_generation_prompt)

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        output_processor = get_output_processor(self._model_name)
        return output_processor.transform(output, inputs)


# Create an instance of the custom IO processor using input and outut processors
# identified/specific by model name
io_processor = _MyInputOutputProcessor(
    model_name=ollama_model_name,
    backend=make_backend("openai", {"model_name": ollama_model_name}),
)

question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]

# With Thinking
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print("------ WITH THINKING ------")
print(">> Thoughts:")
print(outputs.results[0].next_message.reasoning_content)
print(">> Response:")
print(outputs.results[0].next_message.content)
