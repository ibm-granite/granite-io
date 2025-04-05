# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Local
from granite_io import get_output_processor
from granite_io.types import (
    ChatCompletionInputs,
    GenerateResult,
    GenerateResults,
    UserMessage,
)
from tests.test_utils import load_text_file

_GENERAL_MODEL_NAME = "Granite 3.2"
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def test_run_processor_reasoning():
    output_processor = get_output_processor(_GENERAL_MODEL_NAME)
    # Process the output
    results = []
    raw_output = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_raw_reasoning_output.txt")
    )
    results.append(
        GenerateResult(
            completion_string=raw_output,
            completion_tokens=[],
            stop_reason="stop",
        )
    )
    question = (
        "Find the fastest way for a seller to visit all the cities in their region"
    )
    messages = [UserMessage(content=question)]
    outputs = output_processor.transform(
        GenerateResults(results=results),
        ChatCompletionInputs(messages=messages, thinking=True),
    )

    expected_thought_output = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_output_processor_thought.txt")
    )
    assert expected_thought_output == outputs.results[0].next_message.reasoning_content

    expected_response_output = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_output_processor_response.txt")
    )
    assert expected_response_output == outputs.results[0].next_message.content


def test_parse_tool_output():
    # Example from the Granite 3.2 documentation.
    # And yes, this output is wrong. The "location" argument is supposed to be
    # "New York, NY". But this is what the model returns when given this input.
    TOOL_REQUEST = ChatCompletionInputs(
        messages=[
            {"role": "user", "content": "What's the current weather in New York?"}
        ],
        tools=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "arguments": {
                    "location": {
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
            },
            {
                "name": "get_stock_price",
                "description": "Retrieves the current stock price for a given ticker "
                "symbol. The ticker symbol must be a valid symbol for a publicly "
                "traded company on a major US stock exchange like NYSE or NASDAQ. "
                "The tool will return the latest trade price in USD. It should be "
                "used when the user asks about the current or most recent price of "
                "a specific stock. It will not provide any other information about "
                "the stock or company.",
                "arguments": {
                    "ticker": {
                        "description": "The stock ticker symbol, e.g. AAPL for Apple "
                        "Inc."
                    }
                },
            },
        ],
    )
    TOOL_OUTPUT = (
        '<tool_call>[{"name": "get_current_weather", "arguments": '
        '{"location": "New York"}}]'
    )

    output_processor = get_output_processor(_GENERAL_MODEL_NAME)
    outputs = output_processor.transform(
        GenerateResults(
            results=[
                GenerateResult(
                    completion_string=TOOL_OUTPUT,
                    completion_tokens=[],
                    stop_reason="stop",
                )
            ]
        ),
        TOOL_REQUEST,
    )

    # Can't just check the result in one step due to random function IDs
    assert len(outputs.results) == 1
    assert len(outputs.results[0].next_message.tool_calls) == 1
    assert outputs.results[0].next_message.tool_calls[0].name == "get_current_weather"
    assert outputs.results[0].next_message.tool_calls[0].arguments == {
        "location": "New York"
    }


def test_parse_majority_voting_output():
    first_number = 234651
    second_number = 13425
    sum_answer = "248076"
    completion_inputs = ChatCompletionInputs(
        messages=[
            {
                "role": "user",
                "content": f"What is {first_number} + {second_number}?\n"
                f"Answer with just a number please.",
            }
        ],
        majority_voting=True,
        generate_inputs={"n": 10, "temperature": 0.6, "max_tokens": 1024},
    )

    # ruff: noqa: E501
    results_completion_strings = [
        """Here is my thought process:
    The task is to calculate the sum of two numbers, 234651 and 13425.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum, we need to add the two numbers together.

    Here is my response:

    101896

    </answer>""",
        """Here is my thought process:
    The task is to calculate the sum of two numbers, 234651 and 13425. I need to add these numbers together.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    The task is to calculate the sum of two numbers, 234651 and 13425.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum of 234651 and 13425, we need to add these two numbers together.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum of 234651 and 13425, we need to add these two numbers together.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum of 234651 and 13425, we need to add these two numbers together.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum of 234651 and 13425, we need to add these two numbers together.

    Here is my response:

    <answer>348076</answer>""",
        """Here is my thought process:
    The task is to calculate the sum of two numbers, 234651 and 13425.

    Here is my response:

    <answer>248076</answer>""",
        """Here is my thought process:
    To find the sum, we need to add the two numbers together.

    Here is my response:

    101896

    </answer>""",
    ]

    results = []
    for result in results_completion_strings:
        gen_result = GenerateResult(
            completion_string=result,
            completion_tokens=[],
            stop_reason="stop",
        )
        results.append(gen_result)
    output_processor = get_output_processor(_GENERAL_MODEL_NAME)
    outputs = output_processor.transform(
        GenerateResults(results=results),
        completion_inputs,
    )

    assert len(outputs.results) == 1
    assert str(outputs.results[0].next_message.content) == sum_answer
