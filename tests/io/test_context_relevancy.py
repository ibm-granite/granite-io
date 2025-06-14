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
    override_date_for_testing,
)
from granite_io.types import GenerateResult

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "assistant", "content": "Welcome to pet questions!"},
            {"role": "user", "content": "Which of my pets have fleas?"},
        ],
        "documents": [
            {"text": "My dog has fleas."},
            {"text": "My cat does not have fleas."},
            {"text": "My bank account has no money."},
        ],
        "generate_inputs": {
            "temperature": 0.0  # Ensure consistency across runs
        },
    }
)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
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
    You are Granite, developed by IBM.Write the response to the user's input by \
strictly aligning with the facts in the provided documents. If the information needed \
to answer the question is not available in the documents, inform the user that the \
question cannot be answered based on the available data.<|end_of_text|>
    <|start_of_role|>documents<|end_of_role|>Document 0
    My dog has fleas.
    
    Document 1
    My cat does not have fleas.<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Welcome to pet questions!<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Which of my pets have fleas?<|end_of_text|>
    <|start_of_role|>context relevance<|end_of_role|>""")
    assert output == expected_output


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("context relevancy")
    io_proc = ContextRelevancyIOProcessor(backend)

    # Pass our example input thorugh the I/O processor and retrieve the result
    override_date_for_testing(fake_date)  # For consistent VCR output
    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    assert chat_result.results[0].next_message.content in (
        "relevant",
        "irrelevant",
        "partially relevant",
    )
