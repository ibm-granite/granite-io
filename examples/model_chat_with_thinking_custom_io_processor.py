# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model. It uses the framework capability
to create custom IO processor using a specific input processor and output
processor.

In this scenario the chat request enables thinking mode in the model
to provide better understanding of how the model arrived at its answer.
"""

# Local
from granite_io import (
    get_input_processor,
    get_output_processor,
    make_backend,
    make_new_io_processor,
)
from granite_io.types import (
    ChatCompletionInputs,
    UserMessage,
)

ollama_model_name = "granite3.2:8b"
general_model_name = "Granite 3.2"

# Create IO processor using specified input and output processors
io_processor = make_new_io_processor(
    input_processor=get_input_processor(general_model_name),
    output_processor=get_output_processor(general_model_name),
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
