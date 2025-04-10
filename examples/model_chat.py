# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
"""

# Third Party
from dotenv import load_dotenv

# Local
from granite_io import Granite3Point2InputOutputProcessor, OpenAIBackend

# Backends often use environment variables for things like API keys and
# even to allow switching of the API base URL.
# Use a `.env` file to hold environment variables such as OPENAI_API_KEY.
# Refer to ../src/granite_io/backend/README.md for the environment variables
# to use with each type of backend.
# The following command loads those settings into the current environment.
load_dotenv()

# Instantiate a backend for an OpenAI API provider.
# OpenAIBackend() accepts kwargs and env vars like openai.AsyncOpenAI()
backend = OpenAIBackend()

# Create a processor specifically for Granite 3.2 using the backend
processor = Granite3Point2InputOutputProcessor(backend=backend)

# Use a familiar messages structure like OpenAI API.
messages = [
    {
        "role": "user",
        "content": "Find fastest way for a seller to visit all cities in their region",
    },
]

# Additional kwargs for the backend completions.create call can be set here.
# Note:  Don't set `prompt`. The input processor will create that.
completion_kwargs = {
    "model": "granite3.2:2b",
}

# Try a simple completion with the messages and completion kwargs
outputs = processor.create_chat_completion(
    messages=messages,
    generate_inputs=completion_kwargs,
)

print("------ WITHOUT THINKING ANSWER ------")
print(outputs.results[0].next_message.content)

# Additional controls for Granite 3.2 can be set here
controls = {"citations": True}

# Try again using more options
outputs = processor.create_chat_completion(
    messages=messages,
    generate_inputs=completion_kwargs,
    controls=controls,
    thinking=True,
)

print("------ WITH THINKING ANSWER ------")
print(outputs.results[0].next_message.content)
print("------ THOUGHT PROCESS ------")
print(outputs.results[0].next_message.reasoning_content)
