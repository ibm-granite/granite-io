# SPDX-License-Identifier: Apache-2.0

"""
Test cases for query_expansion.py
"""

# Standard
import datetime

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
)
from granite_io.io.query_expansion import QueryExpansionIOProcessor
from granite_io.io.query_rewrite import QueryRewriteIOProcessor

_EXAMPLE_CHAT_INPUT = Granite3Point2Inputs.model_validate(
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
            "temperature": 0.0  # Ensure consistency across runs
        },
    }
)

_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion using the I/O processor.
    """

    backend = lora_server.make_backend()
    query_rewrite_lora_backend = lora_server.make_lora_backend("query_rewrite")

    io_processor = make_io_processor("Granite 3.2", backend=backend)
    rewrite_io_proc = QueryRewriteIOProcessor(query_rewrite_lora_backend)

    rag_io_proc = QueryExpansionIOProcessor(
        backend,
        io_processor,
        rewrite_io_proc,
    )

    qe_result = rag_io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    assert len(qe_result.results) == 5

    # qe_result_strs = [r.next_message.content for r in qe_result.results]
    # print(f"Query Expansion Results: {qe_result_strs}")
