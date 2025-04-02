# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework and an Ollama
backend to serve the model.

It uses majority voting to decide on best answer to use from number of model sample
outputs.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

# By default the backend is an Ollama server running locally using default url. Ollama
# howevere does not support multiple completions per prompt i.e. "n" param below.
# You will need to use a backend that supports multiple completions to fully see
# majority voting happening. Backends like vLLM support multiple completions. You will
# need to update url and key when connecting to another backend.
model_name = "granite3.2:8b"
# openai_base_url = ""
# openai_api_key = ""
io_processor = make_io_processor(
    model_name,
    backend=make_backend(
        "openai",
        {
            "model_name": model_name,
            # "openai_base_url": openai_base_url,
            # "openai_api_key": openai_api_key,
        },
    ),
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
# There will be only 1 output because majority voting is performed
# on all completions results
print(outputs.results[0].next_message.content)
