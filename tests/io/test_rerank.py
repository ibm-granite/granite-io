# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the llm rerank intrinsic.

Note: These tests require the HuggingFace model 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
which will be automatically downloaded on first run (~438MB). Ensure you have internet
connectivity and sufficient disk space. See CONTRIBUTING.md for more details.
"""

# Standard
import os

# Third Party
import pytest
import vcr

# Local
from granite_io import make_io_processor
from granite_io.backend import Backend
from granite_io.backend.openai import OpenAIBackend
from granite_io.io.consts import (
    _GRANITE_3_3_MODEL_NAME,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.llmrerank import RerankRequestProcessor
from granite_io.io.retrieval import (
    InMemoryRetriever,
    RetrievalRequestProcessor,
)

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {
                "role": "assistant",
                "content": "Welcome to the California Appellate Courts help desk.",
            },
            {
                "role": "user",
                "content": "I need to do some legal research to be prepared for my "
                "oral argument. Can I visit the law library?",
            },
        ],
        "generate_inputs": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }
)

_EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


def test_rerank_request_processor(backend_3_3: Backend):  # pylint: disable=redefined-outer-name
    if not isinstance(backend_3_3, OpenAIBackend):
        pytest.skip("Non-OpenAI backends very slow with long requests")
    temp_data_dir = "data/test_retrieval"
    corpus_name = "govt10"
    embeddings_location = f"{temp_data_dir}/{corpus_name}_embeds.parquet"
    if not os.path.exists(embeddings_location):
        raise ValueError(
            f"Embeddings are supposed to be checked into git at "
            f"{embeddings_location} but were not found there."
        )
    io_proc = make_io_processor(_GRANITE_3_3_MODEL_NAME, backend=backend_3_3)
    retriever = InMemoryRetriever(embeddings_location, _EMBEDDING_MODEL_NAME)

    # Enable vcr selectively. The current version of vcrpy crashes the Hugging Face
    # downloader, even when told to ignore all requests to huggingface.co.
    my_vcr = vcr.VCR()
    with my_vcr.use_cassette(
        "tests/io/cassettes/test_rerank/test_rerank_request_processor.yaml"
    ):
        request_processor = RetrievalRequestProcessor(retriever, top_k=128)
        rag_chat_input = request_processor.process(_EXAMPLE_CHAT_INPUT)[0]
        rerank_processor = RerankRequestProcessor(
            io_proc, rerank_top_k=128, return_top_k=128, verbose=True
        )
        rerank_chat_input = rerank_processor.process(rag_chat_input)
    rerank_doc_ids = []
    retrieval_doc_ids = []
    for doc in rerank_chat_input.documents:
        rerank_doc_ids.append(doc.doc_id)
    for doc in rag_chat_input.documents:
        retrieval_doc_ids.append(doc.doc_id)
    match = [
        id1 == id2 for id1, id2 in zip(rerank_doc_ids, retrieval_doc_ids, strict=False)
    ]
    assert not all(match)
