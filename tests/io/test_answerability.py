# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/answerability.py
"""

# Standard
import datetime
import textwrap

# Third Party
import pytest

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.answerability import AnswerabilityIOProcessor
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.types import GenerateResult, GenerateResults

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "assistant", "content": "Welcome to pet questions!"},
            {"role": "user", "content": "Which of my pets have fleas?"},
        ],
        "documents": [
            {"doc_id": 1, "text": "My dog has fleas."},
            {"doc_id": 2, "text": "My cat does not have fleas."},
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
    io_processor = AnswerabilityIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT).prompt
    print("*****")
    print(output)
    print("*****")
    doc1 = '{"document_id": "1"}'
    doc2 = '{"document_id": "2"}'
    expected_output = textwrap.dedent(f"""\
    <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
    Today's Date: {_TODAYS_DATE}.
    You are Granite, developed by IBM. Write the response to the user's input by \
strictly aligning with the facts in the provided documents. If the information needed \
to answer the question is not available in the documents, inform the user that the \
question cannot be answered based on the available data.<|end_of_text|>
    <|start_of_role|>document {doc1}<|end_of_role|>
    My dog has fleas.<|end_of_text|>
    <|start_of_role|>document {doc2}<|end_of_role|>
    My cat does not have fleas.<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Welcome to pet questions!<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Which of my pets have fleas?<|end_of_text|>
    <|start_of_role|>answerability<|end_of_role|>""")
    assert output == expected_output


def test_canned_output():
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = AnswerabilityIOProcessor(None)

    raw_output_to_expected = [
        ("answerable", "answerable"),
        ("unanswerable", "unanswerable"),
        ("<invalid model response>", "ERROR"),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), _EXAMPLE_CHAT_INPUT
        )
        assert len(output.results) == 1
        assert output.results[0].next_message.content == expected

    # Multiple outputs
    multi_raw_output = [
        _make_result(raw_output) for raw_output, _ in raw_output_to_expected
    ]
    multi_expected = [expected for _, expected in raw_output_to_expected]
    multi_output = io_processor.output_to_result(
        GenerateResults(results=multi_raw_output), _EXAMPLE_CHAT_INPUT
    )
    multi_output_strs = [r.next_message.content for r in multi_output.results]
    assert multi_output_strs == multi_expected


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("answerability_prediction")
    io_proc = AnswerabilityIOProcessor(backend)

    # Pass our example input thorugh the I/O processor and retrieve the result
    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    assert chat_result.results[0].next_message.content in ("answerable", "unanswerable")
