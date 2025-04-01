# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.

It uses majority voting to decide on best answer to use from number of
model sample outputs.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
)
question = "What is 234651 + 13425?\nAnswer with just a number please."
messages = [UserMessage(content=question)]
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        majority_voting=True,
        generate_inputs={"n": 20, "temperature": 0.6, "max_tokens": 1024},
    )
)
print("------ WITH MAJORITY VOTING ------")
print(outputs.results[0].next_message.content)
