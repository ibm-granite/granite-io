# SPDX-License-Identifier: Apache-2.0

"""
Test cases for query_rewrite.py
"""

# Standard
import textwrap

# Third Party
import pytest

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.base import RewriteRequestProcessor
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.query_rewrite.query_rewrite import (
    QUERY_TO_REWRITE_TEMPLATE,
    REWRITE_PROMPT_3_3,
    QueryRewriteIOProcessor,
)
from granite_io.types import (
    GenerateResult,
    GenerateResults,
)

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "assistant", "content": "Welcome to pet questions!"},
            {
                "role": "user",
                "content": "I have two pets, a dog named Rex and a cat named Lucy.",
            },
            {
                "role": "assistant",
                "content": "Great, what would you like to share about them?",
            },
            {
                "role": "user",
                "content": "Rex spends a lot of time in the backyard and outdoors, "
                "and Luna is always inside.",
            },
            {
                "role": "assistant",
                "content": "Sounds good! Rex must love exploring outside, while Lucy "
                "probably enjoys her cozy indoor life.",
            },
            {
                "role": "user",
                "content": "But is he more likely to get fleas because of that?",
            },
        ],
        "generate_inputs": {
            "temperature": 0.0,  # Ensure consistency across runs
            "seed": 42,  # Fixed seed for deterministic output
        },
    }
)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
    )


def test_canned_input(_use_fake_date: str):
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    fake_date = _use_fake_date  # Fixture returns the date it put in place
    io_processor = QueryRewriteIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT)

    # Build the expected output using the actual Granite 3.3 format
    query_to_rewrite = QUERY_TO_REWRITE_TEMPLATE.format(
        msg="But is he more likely to get fleas because of that?"
    )

    expected_output = textwrap.dedent(f"""<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: {fake_date}.
You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Welcome to pet questions!<|end_of_text|>
<|start_of_role|>user<|end_of_role|>I have two pets, a dog named Rex and a cat named Lucy.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Great, what would you like to share about them?<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Rex spends a lot of time in the backyard and outdoors, and Luna is always inside.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Sounds good! Rex must love exploring outside, while Lucy probably enjoys her cozy indoor life.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>But is he more likely to get fleas because of that?<|end_of_text|>
{query_to_rewrite}{REWRITE_PROMPT_3_3}""")  # noqa: E501

    print(f"---\n|{output.prompt}|\n---\n")
    print(f"|{expected_output}|\n---")

    assert output.prompt == expected_output


def test_canned_output():
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = QueryRewriteIOProcessor(None)

    raw_output_to_expected = [
        (
            '{ "rewritten_question": "Is Rex more likely to get fleas because he '
            'spends a lot of time in the backyard and outdoors?" }',
            "Is Rex more likely to get fleas because he spends "
            "a lot of time in the backyard and outdoors?",
        ),
        # Current code passes through malformed data unchanged
        ("<invalid model response>", "<invalid model response>"),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), _EXAMPLE_CHAT_INPUT
        )
        print(f"\n{output}\n")

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
    print(f"\n{output}\n")
    assert multi_output_strs == multi_expected


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("query_rewrite")
    io_proc = QueryRewriteIOProcessor(backend)

    # Pass our example input through the I/O processor and retrieve the result
    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    print(f"\n{chat_result.results[0].next_message.content}\n")
    assert (
        chat_result.results[0].next_message.content
        == "Is Rex more likely to get fleas because he spends "
        "a lot of time in the backyard and outdoors?"
    )


@pytest.mark.vcr
def test_request_processor(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using a RequestProcessor
    """
    backend = lora_server.make_lora_backend("query_rewrite")
    io_proc = QueryRewriteIOProcessor(backend)
    request_proc = RewriteRequestProcessor(io_proc)

    # Pass our example input through the rewrite and retrieve the result
    rewrites = request_proc.process(_EXAMPLE_CHAT_INPUT)
    assert len(rewrites) == 1
    rewritten_input = rewrites[0]

    assert len(rewritten_input.messages) == len(_EXAMPLE_CHAT_INPUT.messages)
    assert (
        rewritten_input.messages[-1].content
        == "Is Rex more likely to get fleas because he spends "
        "a lot of time in the backyard and outdoors?"
    )
