# SPDX-License-Identifier: Apache-2.0

"""
This example show how to infer or call a model using the framework
and an Ollama backend to serve the model.
In this scenario, it is a simple text from Wikipedia use case where
citations are enabled.
"""

# Standard
import pprint

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
)
question = "What team is the most successful hurling team in the last 10 years?"
messages = [UserMessage(content=question)]

# Source: https://en.wikipedia.org/wiki/Limerick_county_hurling_team
documents = [
    {
        "text": "The 2018 season concluded with Limerick winning the 2018 All-Ireland SHC, the team's first since 1973, with a 3–16 to 2–18 point defeat of Galway in the final.The team built on this success, winning the NHL in 2019, 2020 and 2023, the Munster SHC in 2019, 2020, 2021 and 2022 and the All-Ireland SHC again in 2020, 2021 and 2022. Munster Senior Hurling Championship 2023, All Ireland Hurling Championship 2023 to be forever remembered the team to join the Cork hurling Champions of the 40s and the Kilkenny hurling Champions of the 2000s to complete 4 in a row."  # noqa: E501
    },
]

# With RAG and citations
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        documents=documents,
        controls={"citations": True},
    )
)
print("\n\n>> Model raw output:\n")
response = outputs.results[0].next_message.raw
print(response)

print("\n\n>> Response:\n")
response = outputs.results[0].next_message.content
print(response)

if outputs.results[0].next_message.citations:
    print("\n\n>> Citations:\n")
    pprint.pprint(outputs.results[0].next_message.citations, sort_dicts=False)

    print("\n\n>> Documents:\n")
    pprint.pprint(outputs.results[0].next_message.documents, sort_dicts=False)

if outputs.results[0].next_message.hallucinations:
    print("\n\n>> Hallucinations:\n")
    pprint.pprint(outputs.results[0].next_message.hallucinations, sort_dicts=False)
