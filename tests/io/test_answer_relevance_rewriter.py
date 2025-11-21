# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/answer_relevancy.py
"""

# Standard
import datetime
import json
import logging
import pathlib
import textwrap

# Third Party
import pytest
import yaml

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.answer_relevance import (
    AnswerRelevanceRewriteIOProcessor,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.types import GenerateResult, GenerateResults
from conftest import answer_relevance_rewriter_lora_exists_locally

logging.basicConfig(level=logging.INFO)
# logging.getLogger("vcr").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
    )


_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")


DATA_FILE = pathlib.Path(__file__).parent / "data" / "test_answer_relevance3.yaml"

with DATA_FILE.open() as f:
    ALL_DATA = yaml.safe_load(f)

examples = ALL_DATA["examples"]

# Process the example data to populate all derived data for the tests.
for example in examples:
    # Rewriter
    classifier_judgment = example["classifier_output"]["answer_relevance_judgment"]
    if classifier_judgment:
        example["rewriter_input"] = None
        example["rewriter_prompt"] = None
        example["rewriter_output_str"] = None
    else:
        classifier_input_messages = example["classifier_input"]["messages"]
        rewriter_input_messages = classifier_input_messages + [
            {"role": "user", "content": "answer_relevance"},
            {"role": "assistant", "content": json.dumps(example["classifier_output"])},
        ]
        example["rewriter_input"] = {
            "messages": rewriter_input_messages,
            "documents": example["classifier_input"]["documents"],
            "generate_inputs": example["classifier_input"]["generate_inputs"],
        }
        example["rewriter_prompt"] = example["rewriter_prompt"].format(
            _TODAYS_DATE=_TODAYS_DATE
        )
        example["rewriter_output_str"] = json.dumps(example["rewriter_output"])
        example["rewriter_input"] = Granite3Point3Inputs.model_validate(
            example["rewriter_input"]
        )


@pytest.mark.parametrize(
    "case",
    examples,
    ids=lambda c: c["name"],
)
def test_rewriter_canned_input(case):
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    logger.info("test_rewriter_canned_input")
    io_processor = AnswerRelevanceRewriteIOProcessor(None)
    rewriter_input = case["rewriter_input"]
    if not rewriter_input:
        return
    output = io_processor.inputs_to_generate_inputs(rewriter_input).prompt
    logger.info(output)
    expected_output = textwrap.dedent(case["rewriter_prompt"])
    assert output == expected_output


@pytest.mark.parametrize(
    "case",
    examples,
    ids=lambda c: c["name"],
)
def test_rewriter_canned_output(case):
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = AnswerRelevanceRewriteIOProcessor(None)

    rewriter_input = case["rewriter_input"]
    if not rewriter_input:
        return
    rewriter_output = case["rewriter_output"]
    rewriter_output_str = case["rewriter_output_str"]
    raw_output_to_expected = [
        (rewriter_output_str, rewriter_output),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), rewriter_input
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
        GenerateResults(results=multi_raw_output), rewriter_input
    )
    multi_output_strs = [r.next_message.content for r in multi_output.results]
    multi_output_jsons = [json.loads(r) for r in multi_output_strs]
    assert multi_output_jsons == multi_expected


@pytest.mark.skipif(not answer_relevance_rewriter_lora_exists_locally,
                    reason = "Missing local lora path")
@pytest.mark.vcr(
    record_mode="none"  # none, all, once, new_episodes
    # record_mode="new_episodes"
)
@pytest.mark.parametrize("case", examples, ids=lambda c: c["name"])
def test_run_rewriter(case, lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    logger.info("test_run_rewriter")
    backend = lora_server.make_lora_backend("answer_relevance_rewriter")
    io_proc = AnswerRelevanceRewriteIOProcessor(backend)

    # override_date_for_testing(_use_fake_date)  # For consistent VCR output
    rewriter_input = case["rewriter_input"]
    if not rewriter_input:
        return
    expected = case["rewriter_output"]
    chat_result = io_proc.create_chat_completion(rewriter_input)
    actual = chat_result.results[0].next_message.content
    assert actual == expected


def _print_messages(messages):
    return "\n".join([f"{message.role}: {message.content}" for message in messages])
