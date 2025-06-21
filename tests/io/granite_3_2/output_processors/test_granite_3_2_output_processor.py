# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
from test_utils import load_text_file

# Local
from granite_io import get_output_processor
from granite_io.types import (
    ChatCompletionInputs,
    GenerateResult,
    GenerateResults,
    UserMessage,
)

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
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for, "
                            'in the format "City, State, Country".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit to return the temperature in. "
                            'Defaults to "celsius".',
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_stock_price",
                "description": "Retrieve lowest and highest stock price for a given "
                "stock symbol and date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol, for example MSFT for "
                            "Microsoft",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format",
                        },
                    },
                    "required": ["symbol", "date"],
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
