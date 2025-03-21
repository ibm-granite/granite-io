# SPDX-License-Identifier: Apache-2.0

"""
Tests for the majority voting I/O processor
"""

# Standard
import re

# Third Party
import pytest

# Local
from granite_io import (
    get_input_processor,
    get_output_processor,
    make_backend,
    make_new_io_processor,
)
from granite_io.backend import Backend
from granite_io.types import ChatCompletionInputs, ChatCompletionResults

ollama_model_name = "granite3.2:8b"
general_model_name = "Granite 3.2"


# All the different chat completion requests that are tested in this file, serialized as
# JSON strings. Represented as a dictionary instead of a list so that pytest output will
# show the short key instead of the long value when referencing a single run of a test
INPUT_JSON_STRS = {
    "simple": """
{
    "messages":
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"}
    ]
}
""",
    "thinking_tag": """
{
    "messages":
    [
        {"role": "user", "content": "How much wood could a wood chuck chuck?"}
    ],
    "thinking": true
}
""",
    "stop_strings": """
{
    "messages":
        [
            {"role": "user", "content": "How much wood could a wood chuck chuck?"}
        ],
    "generate_inputs":
        {
            "stop": "woodchuck"
        }
}
""",
}


@pytest.fixture(scope="session", params=INPUT_JSON_STRS)
def input_json_str(request: pytest.FixtureRequest) -> str:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different chat completion requests."""
    return INPUT_JSON_STRS[request.param]


@pytest.mark.vcr
def test_make_io_processor(backend_x: Backend, input_json_str: str):  # pylint: disable=redefined-outer-name
    inputs = ChatCompletionInputs.model_validate_json(input_json_str)
    io_processor = make_new_io_processor(
        input_processor=get_input_processor(general_model_name),
        output_processor=get_output_processor(general_model_name),
        backend=backend_x,
    )
    outputs: ChatCompletionResults = io_processor.create_chat_completion(inputs)

    assert isinstance(outputs, ChatCompletionResults)
    assert len(outputs.results) == 1

    content = outputs.results[0].next_message.content
    assert content  # Make sure we don't get empty result

    # Test for stop reason
    if inputs.generate_inputs and inputs.generate_inputs.stop:
        stop = inputs.generate_inputs.stop
        # Note: currently transformers includes the stop tokens,
        # but OpenAI does not. So, either endswith or not-in.
        assert stop not in content
        assert outputs.results[0].next_message.stop_reason == "stop"

    # TODO: Verify outputs in greater detail


def test_make_io_processor_no_input_processor():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attempted to create IO Processor without setting an Input Processor"
        ),
    ):
        _ = make_new_io_processor(
            input_processor=None,
            output_processor=get_output_processor(general_model_name),
            backend=make_backend("openai", {"model_name": ollama_model_name}),
        )


def test_make_io_processor_no_output_processor():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attempted to create IO Processor without setting an Output Processor"
        ),
    ):
        _ = make_new_io_processor(
            input_processor=get_input_processor(general_model_name),
            output_processor=None,
            backend=make_backend("openai", {"model_name": ollama_model_name}),
        )
