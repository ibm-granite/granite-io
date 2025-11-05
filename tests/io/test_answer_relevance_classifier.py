# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/answer_relevancy.py
"""

# Standard
import datetime
import textwrap
import json
import logging

# Third Party
import pytest

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.answer_relevance import (
    AnswerRelevanceIOProcessor,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
    override_date_for_testing,
)
from granite_io.types import GenerateResult, GenerateResults

logging.basicConfig(level=logging.INFO)
# logging.getLogger("vcr").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
    )


_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")

import pathlib
import pytest
import yaml

DATA_FILE = pathlib.Path(__file__).parent / "data" / "test_answer_relevance3.yaml"

with DATA_FILE.open() as f:
    ALL_DATA = yaml.safe_load(f)

examples = ALL_DATA["examples"]

# Process the example data to populate all derived data for the tests.
for example in examples:
    # Classifier
    example["classifier_prompt"] = example["classifier_prompt"].format(
        _TODAYS_DATE=_TODAYS_DATE
    )
    example["classifier_output_str"] = json.dumps(example["classifier_output"])

    example["classifier_input"] = Granite3Point3Inputs.model_validate(
        example["classifier_input"]
    )


@pytest.mark.parametrize(
    "case",
    examples,
    ids=lambda c: c["name"],
)
def test_classifier_canned_input(case):
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    io_processor = AnswerRelevanceIOProcessor(None)
    input = case["classifier_input"]
    logger.info(input)
    output = io_processor.inputs_to_generate_inputs(input).prompt
    logger.info(output)
    expected_output = textwrap.dedent(case["classifier_prompt"])
    assert output == expected_output


@pytest.mark.parametrize(
    "case",
    examples,
    ids=lambda c: c["name"],
)
def test_classifier_canned_output(case):
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = AnswerRelevanceIOProcessor(None)

    classifier_input = case["classifier_input"]
    classifier_output = case["classifier_output"]
    classifier_output_str = case["classifier_output_str"]
    raw_output_to_expected = [
        (
            classifier_output_str,
            classifier_output,
        )
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), classifier_input
        )
        assert len(output.results) == 1
        actual = output.results[0].next_message.content
        actual = json.loads(actual)
        assert actual == expected

    # Multiple outputs
    multi_raw_output = [
        _make_result(raw_output) for raw_output, _ in raw_output_to_expected
    ]
    multi_expected = [expected for _, expected in raw_output_to_expected]
    multi_output = io_processor.output_to_result(
        GenerateResults(results=multi_raw_output), classifier_input
    )
    multi_output_strs = [r.next_message.content for r in multi_output.results]
    multi_output_jsons = [json.loads(r) for r in multi_output_strs]
    assert multi_output_jsons == multi_expected


@pytest.mark.vcr(
    record_mode="none"
    # record_mode="new_episodes"
)
@pytest.mark.parametrize("case", examples, ids=lambda c: c["name"])
def test_run_classifier(case, lora_server: LocalVLLMServer, fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    name = case["name"]
    backend = lora_server.make_lora_backend("answer_relevance_classifier")
    io_proc = AnswerRelevanceIOProcessor(backend)

    override_date_for_testing(fake_date)  # For consistent VCR output
    classifier_input = case["classifier_input"]
    classifier_output = case["classifier_output"]

    logger.info(
        f"== Classifier input messages\n{_print_messages(classifier_input.messages)}"
    )

    chat_result = io_proc.create_chat_completion(classifier_input)

    output_json = json.loads(chat_result.results[0].next_message.content)
    logger.info(
        f"== Classifier output content\n{json.dumps(output_json, indent=2)}"
    )

    assert output_json == classifier_output


def _print_messages(messages):
    return "\n".join([f"{message.role}: {message.content}" for message in messages])
