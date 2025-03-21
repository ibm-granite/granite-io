# SPDX-License-Identifier: Apache-2.0

"""
This example show how to process input to a model(prompt) and putput from the model
using the framework.

The inference of the model is managed independent of the framework using Ollama backend
to serve the model.

In this scenario the chat request enables thinking mode in the model to provide better
understanding of how the model arrived at its answer.
"""

# Third Party
import openai

# Local
from granite_io import get_input_processor, get_output_processor
from granite_io.types import (
    ChatCompletionInputs,
    GenerateResult,
    GenerateResults,
    UserMessage,
)

api_key = "ollama"
base_url = "http://localhost:11434/v1"  # Ollama server hosting model
ollama_model_name = "granite3.2:8b"

# Connect to Ollama server
openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)

# Get input processor for the model type and version
input_processor = get_input_processor(
    ollama_model_name,
)

# Construct the prompt for Granite 3.2 model so that it will provide reasoning or
# thinking of how it provided its answer to the question asked. It will be similar
# to this format:
#
# prompt = """<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
# Today's Date: March 13, 2025.
# You are Granite, developed by IBM. You are a helpful AI assistant.
# Respond to every user query in a comprehensive and detailed way. You can write down \
# your thoughts and reasoning process before responding. In the thought process, \
# engage in a comprehensive cycle of analysis, summarization, exploration, \
# reassessment, reflection, backtracing, and iteration to develop well-considered \
# thinking process. In the response section, based on various attempts, explorations, \
# and reflections from the thoughts section, systematically present the final \
# solution that you deem correct. The response should summarize the thought process. \
# Write your thoughts after 'Here is my thought process:' and write your response \
# after 'Here is my response:' for each user query.<|end_of_text|>
# <|start_of_role|>user<|end_of_role|>Find the fastest way for a seller to visit all \
# the cities in their region<|end_of_text|>
# <|start_of_role|>assistant<|end_of_role|>"""
question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]
prompt = input_processor.transform(
    ChatCompletionInputs(messages=messages, thinking=True)
)

# Infer/query the model
result = openai_client.completions.create(
    model=ollama_model_name,
    prompt=prompt,
)

# Process the output as standard
print("\n\n *** Standard Output ***\n\n")
for choice in result.choices:
    print(f"{choice.text}\n")

# Process output using granite-io output processor. Output processor parses the output
# into response and reasoning for you

# Get output processor for the model type and version
output_processor = get_output_processor(
    ollama_model_name,
)

# Process the output
results = []
for choice in result.choices:
    results.append(
        GenerateResult(
            completion_string=choice.text,
            completion_tokens=[],  # Not part of the OpenAI spec
            stop_reason=choice.finish_reason,
        )
    )
outputs = output_processor.transform(
    GenerateResults(results=results),
    ChatCompletionInputs(messages=messages, thinking=True),
)

print("\n\n *** granite-io Output ***\n\n")
print(">> Thoughts:")
print(outputs.results[0].next_message.reasoning_content)
print("\n>> Response:")
print(outputs.results[0].next_message.content)
