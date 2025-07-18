# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/context_relevancy.py
"""

# Standard
import datetime
import textwrap

# Third Party
import pytest

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.context_relevancy import ContextRelevancyIOProcessor
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "assistant", "content": "Welcome to pet questions!"},
            {"role": "user", "content": "Which of my pets have fleas?"},
        ],
        "documents": [
            {"doc_id": 1, "text": "My dog has fleas."},
        ],
        "generate_inputs": {
            "temperature": 0.0  # Ensure consistency across runs
        },
    }
)

_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")


def test_canned_input():
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    io_processor = ContextRelevancyIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT).prompt
    print("*****")
    print(output)
    print("*****")
    expected_output = textwrap.dedent(f"""\
    <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
    Today's Date: {_TODAYS_DATE}.
    You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Welcome to pet questions!<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Which of my pets have fleas?<|end_of_text|>
    <|start_of_role|>final_user_query<|end_of_role|>Which of my pets have \
fleas?<|end_of_text|>
    <|start_of_role|>document {{"document_id": "1"}}<|end_of_role|>
    My dog has fleas.<|end_of_text|>
    <|start_of_role|>context_relevance: Analyze the provided document in relation to \
the final user query from the conversation. Determine if the document contains \
information that could help answer the final user query. Output 'relevant' if the \
document contains substantial information directly useful for answering the final \
user query. Output 'partially relevant' if the document contains some related \
information that could partially help answer the query, or if you are uncertain \
about the relevance - err on the side of 'partially relevant' when in doubt. \
Output 'irrelevant' only if the document clearly contains no information that \
could help answer the final user query. When uncertain, choose 'partially \
relevant' rather than 'irrelevant'. Your output should be a JSON structure with \
the context relevance classification:
    ```json
    {{
        "context_relevance": "YOUR_CONTEXT_RELEVANCE_CLASSIFICATION_HERE"
    }}
    ```<|end_of_role|>""")
    assert output == expected_output


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("context_relevancy")
    io_proc = ContextRelevancyIOProcessor(backend)

    # Pass our example input thorugh the I/O processor and retrieve the result
    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    assert chat_result.results[0].next_message.content == "relevant"
