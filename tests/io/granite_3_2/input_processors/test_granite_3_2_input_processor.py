# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Local
from granite_io import get_input_processor
from granite_io.types import (
    ChatCompletionInputs,
    UserMessage,
)
from tests.test_utils import load_text_file

_GENERALE_MODEL_NAME = "Granite 3.2"
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def test_run_processor_reasoning():
    input_processor = get_input_processor(_GENERALE_MODEL_NAME)
    question = (
        "Find the fastest way for a seller to visit all the cities in their region"
    )
    messages = [UserMessage(content=question)]
    prompt = input_processor.transform(
        ChatCompletionInputs(messages=messages, thinking=True)
    )

    expected_prompt = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_prompt.txt")
    )
    assert isinstance(prompt, str)
    assert len(prompt) == len(expected_prompt)

    # Prompt contains dates in first two lines of the text which change.
    # Therefore need to extract and test them separately first.
    # Then we test the remaining text.
    assert (
        prompt.split("\n", 1)[0].split(":")[0]
        == (expected_prompt.split("\n", 1)[0].split(":")[0])
    )
    assert (
        prompt.split("\n", 1)[1].split(":")[0]
        == (expected_prompt.split("\n", 1)[1].split(":")[0])
    )
    assert prompt.split("\n", 2)[-1] == expected_prompt.split("\n", 2)[-1]


def test_run_processor_majority_voting():
    input_processor = get_input_processor(_GENERALE_MODEL_NAME)
    question = "What is 234651 + 13425?\nAnswer with just a number please."
    messages = [UserMessage(content=question)]
    prompt = input_processor.transform(
        ChatCompletionInputs(messages=messages, majority_voting=True)
    )

    expected_prompt = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_majority_voting_prompt.txt")
    )
    assert isinstance(prompt, str)
    assert len(prompt) == len(expected_prompt)

    # Prompt contains dates in first two lines of the text which change.
    # Therefore need to extract and test them separately first.
    # Then we test the remaining text.
    assert (
        prompt.split("\n", 1)[0].split(":")[0]
        == (expected_prompt.split("\n", 1)[0].split(":")[0])
    )
    assert (
        prompt.split("\n", 1)[1].split(":")[0]
        == (expected_prompt.split("\n", 1)[1].split(":")[0])
    )
    assert prompt.split("\n", 2)[-1] == expected_prompt.split("\n", 2)[-1]
