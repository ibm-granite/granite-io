# SPDX-License-Identifier: Apache-2.0

"""
This example show how to use sequential inference scaling
to enforce the that model response has no hallucination.
"""

# Third Party
from granite_io_pdl.sequential_scaling import SequentialScalingInputOutputProcessor

# Local
from granite_io.types import ChatCompletionInputs, ChatCompletionResults, UserMessage


def hallucination_validator(processor_outputs: ChatCompletionResults):
    return not processor_outputs.results[0].next_message.hallucinations


model_name = "granite3.2:8b"
io_processor = SequentialScalingInputOutputProcessor(
    model=model_name, backend="openai", validator=hallucination_validator
)
question = "What is RAG?"
messages = [UserMessage(content=question)]

documents = [
    {
        "text": "Dynamic Data: Automatically generate and refine graphs to best fit your domain and ontology needs."  # noqa: E501
    },
    {
        "text": "RAG, retrieval-augmented generation, is a technique that grants generative artificial intelligence models information retrieval capabilities."  # noqa: E501
    },
]

outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        documents=documents,
        controls={"hallucinations": True},
    )
)
print("\n\n>> Response:\n")
response = outputs.results[0].next_message.content
print(response)

print("\n\n>> Hallucinations:\n")
if outputs.results[0].next_message.hallucinations:
    print("We are sorry, we tried to not hallucinate!")
else:
    print("We are happy to tell you that we did not hallucinate!")
