# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/answer_relevancy.py
"""

# Standard
import datetime
import logging
import pathlib

# Third Party
import pytest
import yaml

# Local
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.answer_relevance import (
    AnswerRelevanceCompositeIOProcessor,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
    # override_date_for_testing,
)
from granite_io.types import GenerateResult

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
    example["classifier_input"] = Granite3Point3Inputs.model_validate(
        example["classifier_input"]
    )


@pytest.mark.vcr(
    record_mode="none"  # none, all, once, new_episodes
    # record_mode="new_episodes"
)
@pytest.mark.parametrize("case", examples, ids=lambda c: c["name"])
def test_run_composite(case, lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    classifier_backend = lora_server.make_lora_backend("answer_relevance_classifier")
    rewriter_backend = lora_server.make_lora_backend("answer_relevance_rewriter")
    io_proc = AnswerRelevanceCompositeIOProcessor(classifier_backend, rewriter_backend)

    # override_date_for_testing(_use_fake_date)  # For consistent VCR output

    classifier_input = case["classifier_input"]
    output = case["composite_output"]
    logger.info(
        f"Test composite input messages\n{_print_messages(classifier_input.messages)}"
    )

    chat_result = io_proc.create_chat_completion(classifier_input)
    chat_result_message_content = chat_result.results[0].next_message.content

    logger.info(
        "Test composite output message\n%s", chat_result.results[0].next_message
    )
    logger.info(f"Test composite output message content\n{chat_result_message_content}")

    actual = chat_result_message_content
    logger.info(f"Test composite output json\n{actual}")

    assert actual == output


def _print_messages(messages):
    return "\n".join([f"{message.role}: {message.content}" for message in messages])
