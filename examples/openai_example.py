# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
"""

# Standard
import os

# Local
from granite_io import (
    Granite3Point2InputOutputProcessor,
    Granite3Point2Inputs,
    OpenAIBackend,
)

# TODO: dotenv
apikey = os.environ.get("OPENAI_API_KEY")
model_name = os.environ.get("MODEL_NAME")


backend = OpenAIBackend(default_headers={"RITS_API_KEY": apikey})
processor = Granite3Point2InputOutputProcessor(backend=backend)

messages = [
    {
        "role": "user",
        "content": "Find fastest way for a seller to visit all cities in their region",
    },
]

generate_inputs = {
    "model": model_name,
}

inputs = Granite3Point2Inputs(messages=messages, generate_inputs=generate_inputs)

outputs = processor.create_chat_completion(inputs)

print("------ WITHOUT THINKING ------")
print(outputs.results[0].next_message.content)
