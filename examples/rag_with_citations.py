# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
In this scenario, it is a simple RAG use case where citations are enabled.
"""

# Standard
import pprint

# Local
from granite_io import make_backend, make_io_processor
from granite_io.io.model_output_parser import parse_model_output
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
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

# With RAG and citations
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages, documents=documents, controls={"citations": True}
    )
)
print(">> Response:\n\n")
response = outputs.results[0].next_message.content
print(response)

print("\n\n>> Extract response into text, citation and hallucination constiuents:\n\n")
parsed_output = parse_model_output(response)
pprint.pprint(parsed_output, sort_dicts=False)
