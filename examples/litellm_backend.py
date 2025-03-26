# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an litellm->Ollama backend to serve the model.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "ollama/granite3.2:8b"
io_processor = make_io_processor(
    "Granite 3.2", backend=make_backend("litellm", {"model_name": model_name})
)
question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print("------ WITHOUT THINKING ------")
print(outputs.results[0].next_message.content)

print("------ TRY AGAIN ------")
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print("------ AGAIN RESULTS ------")
print(outputs.results[0].next_message.content)
