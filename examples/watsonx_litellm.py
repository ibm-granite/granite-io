# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and a watsonx or ollama backend with the LiteLLM client.
"""

# Third Party
from dotenv import load_dotenv

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

load_dotenv()

# LiteLLM will honor environment variables.
# For this example, use a local .env file to set the variables needed
# for watsonx or for ollama.  load_dotenv() above will read them in.
#
# Example .env entries:
#
# For watsonx:
#
# MODEL_NAME=watsonx/ibm/granite-3-2-8b-instruct
# WATSONX_API_BASE=https://us-south.ml.cloud.ibm.com
# WATSONX_PROJECT_ID=<your watsonx project ID>
# WATSONX_API_KEY=<your watsonx api key>
#
# For ollama:
#
# MODEL_NAME="ollama/granite3.2:8b"
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_API_KEY="ollama"

model_name = "Granite 3.2"
io_processor = make_io_processor(
    model_name,
    backend=make_backend(
        "litellm",
        {
            "model_name": "ollama/granite3.2:8b",
        },
    ),
)
question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print("------ WITHOUT THINKING ------")
print(outputs.results[0].next_message.content)

# With Thinking
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print("------ WITH THINKING ------")
print(">> Thoughts:")
print(outputs.results[0].reasoning_content)
print(">> Response:")
print(outputs.results[0].next_message.content)
