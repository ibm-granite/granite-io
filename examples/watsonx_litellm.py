# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and a watsonx, Replicate, or Ollama backend with the LiteLLM client.
In this scenario the chat request enables thinking mode in the model
to provide better understanding of how the model arrived at its answer.
"""

# Third Party
from dotenv import load_dotenv

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

load_dotenv()

# LiteLLM will honor environment variables.
# For this example, use a local .env file to set the variables needed
# for watsonx or Replicate or Ollama.  load_dotenv() above will read them in.
#
# Example .env entries (choose one provider):
#
# For watsonx:
#
# MODEL_NAME=watsonx/ibm/granite-3-2-8b-instruct
# WATSONX_API_BASE=https://us-south.ml.cloud.ibm.com
# WATSONX_PROJECT_ID=<your watsonx project ID>
# WATSONX_API_KEY=<your watsonx api key>
#
# For Replicate:
#
# REPLICATE_API_TOKEN=<your Replicate api key>
# MODEL_NAME=replicate/ibm-granite/granite-3.2-8b-instruct

# For Ollama:
#
# MODEL_NAME="ollama/granite3.2:8b"
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_API_KEY="ollama"

# NOTE! If you don't specify any of the above, the Ollama defaults are used.

model_type = "Granite 3.2"
model_name = "ollama/granite3.2:8b"
io_processor = make_io_processor(
    model_type, backend=make_backend("litellm", {"model_name": model_name})
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
