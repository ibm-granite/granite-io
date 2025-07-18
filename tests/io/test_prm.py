# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Granite PRM intrinsic's I/O processor
"""

# Standard
import datetime
import textwrap

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.process_reward_model.best_of_n import (
    PRMBestOfNCompositeIOProcessor,
    ProcessRewardModelIOProcessor,
)

_EXAMPLE_CHAT_INPUT_CORRECT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "user", "content": "What is the square of 5+1?"},
            {
                "role": "assistant",
                "content": "To find the square of 5+1, we first calculate 5+1=6.\n "
                "The square of 6 is 36.",
            },
        ],
        "generate_inputs": {"temperature": 0.0},
    }
)

_EXAMPLE_CHAT_INPUT_INCORRECT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "user", "content": "What is the square of 5+1?"},
            {
                "role": "assistant",
                "content": "To find the square of 5+1, we first calculate 5+1=50.\n "
                "The square of 50 is 100.",
            },
        ],
        "generate_inputs": {"temperature": 0.0},
    }
)

_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")


def test_input():
    """
    Validates that the I/O Processor handles the input correctly
    """
    io_processor = ProcessRewardModelIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT_CORRECT).prompt
    print(f"Actual output:\n{output}")
    expected_output = textwrap.dedent(f"""\
    <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
    Today's Date: {_TODAYS_DATE}.
    You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>What is the square of 5+1? \
To find the square of 5+1, we first calculate 5+1=6. \
Is this response correct so far (Y/N)?<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Y<|end_of_text|>
    <|start_of_role|>user<|end_of_role|> The square of 6 is 36. \
Is this response correct so far (Y/N)?<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Y<|end_of_text|>
    """)
    assert output.strip() == expected_output.strip()


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("prm")
    io_proc = ProcessRewardModelIOProcessor(backend)

    # Pass our example input through the I/O processor and retrieve the result
    good_chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT_CORRECT)
    # good chat should have a high score
    assert float(good_chat_result.results[0].next_message.content) > 0.9

    bad_chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT_INCORRECT)
    # bad chat should have a low score
    assert float(bad_chat_result.results[0].next_message.content) < 0.7


@pytest.mark.vcr
def test_run_composite(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Generate chat completions and check certainty using a composite I/O processor to
    choreograph the flow.
    """
    granite_backend = lora_server.make_backend()
    lora_backend = lora_server.make_lora_backend("prm")
    granite_io_proc = make_io_processor("Granite 3.3", backend=granite_backend)
    io_proc = PRMBestOfNCompositeIOProcessor(
        granite_io_proc,
        lora_backend,
    )

    # Check that the process returns one final answer
    input_without_msg = _EXAMPLE_CHAT_INPUT_CORRECT.model_copy(
        update={"messages": _EXAMPLE_CHAT_INPUT_CORRECT.messages[:-1]}
    ).with_addl_generate_params({"temperature": 1.0, "n": 5, "max_tokens": 2048})
    results = io_proc.create_chat_completion(input_without_msg)
    assert len(results.results) == 1
    assert "36" in results.results[0].next_message.content
