# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
"""

# Standard
import os

# Local
from granite_io import Granite3Point2InputOutputProcessor, LiteLLMBackend

# TODO: dotenv
# apikey = os.environ.get("OPENAI_API_KEY")
model_name = os.environ.get("MODEL_NAME")

backend = LiteLLMBackend()

processor = Granite3Point2InputOutputProcessor(backend=backend)

messages = [
    {
        "role": "user",
        "content": "Find fastest way for a seller to visit all cities in their region",
    },
]

generate_inputs = {
    "model": model_name,
    "temperature": 0.8,
}

controls = {"citations": True}

# inputs = Granite3Point2Inputs(messages=messages, generate_inputs=generate_inputs)
# inputs = {"messages": messages, "generate_inputs": generate_inputs}
# outputs = processor.create_chat_completion(**inputs)
outputs = processor.create_chat_completion(
    messages=messages,
    generate_inputs=generate_inputs,
)

print("------ WITHOUT THINKING ------")
print(outputs.results[0].next_message.content)

outputs = processor.create_chat_completion(
    messages=messages,
    generate_inputs=generate_inputs,
    controls=controls,
    thinking=True,
)

print("------ WITH THINKING ------")
print(outputs.results[0].next_message.content)
