# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
In this scenario, it is a simple RAG use case where citations are enabled.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.io.model_output_parser import parse_output
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
print(">> Response:")
response = outputs.next_message.content
print(response)

print(">> Extract response into text and citation categories:")
response_separated_by_citation = outputs.next_message.content.split("# Citations:")
if len(response_separated_by_citation) == 2:
    parse_output(response, response_separated_by_citation[1])
else:
    raise RuntimeError(f"""Unable to separate reponse by citation:
                       {response_separated_by_citation}""")
