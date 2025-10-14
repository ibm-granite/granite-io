# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the retrieval intrinsic.

Note: These tests require the HuggingFace model
'sentence-transformers/multi-qa-mpnet-base-dot-v1' which will be automatically
downloaded on first run (~438MB). Ensure you have internet connectivity and
sufficient disk space. See CONTRIBUTING.md for more details.
"""

# Standard
from unittest import mock
import pathlib
import tempfile

# Third Party
import pytest
import torch

# Local
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
)
from granite_io.io.retrieval import (
    ElasticsearchRetriever,
    InMemoryRetriever,
    RetrievalRequestProcessor,
    compute_embeddings,
    write_embeddings,
)
from granite_io.io.retrieval.util import (
    read_mtrag_corpus,
)

_EXAMPLE_CHAT_INPUT = Granite3Point2Inputs.model_validate(
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

_DATA_DIR = pathlib.Path("data/test_retrieval")
_EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


@pytest.fixture
def govt_embeddings_file():
    """
    :returns: a pre-indexed copy of a tiny slice of the MTRAG benchmark's "govt" data
     set.
    """
    target_file = _DATA_DIR / "govt10_embeds.parquet"
    return target_file


@pytest.fixture
def govt_docs_file():
    """
    :returns: a copy of a tiny slice of the MTRAG benchmark's "govt" data set.
    """
    target_file = _DATA_DIR / "govt10.jsonl.zip"
    return target_file


def test_make_embeddings(govt_docs_file):  # pylint: disable=redefined-outer-name
    """
    Verify that embedding creation is working by creating embeddings for the first
    10 documents in the "govt" corpus from the MTRAG benchmark.
    """
    full_govt = read_mtrag_corpus(govt_docs_file)
    corpus = full_govt.slice(0, 10)
    embeddings = compute_embeddings(corpus, _EMBEDDING_MODEL_NAME)
    assert embeddings.column("embedding").to_pylist()[0][:10] == pytest.approx(
        [
            -0.11038033664226532,
            0.27693408727645874,
            -0.11863572895526886,
            -0.0792723074555397,
            0.20247098803520203,
            0.09491363912820816,
            0.6091732978820801,
            0.0905776172876358,
            0.10194987803697586,
            -0.011982650496065617,
        ],
        abs=1e-3,
    )

    # Round-trip through a file and make sure we get the embeddings back.
    with tempfile.TemporaryDirectory() as tmpdir:
        file_loc = write_embeddings(tmpdir, "test", embeddings)
        retriever = InMemoryRetriever(file_loc, _EMBEDDING_MODEL_NAME)
        # pylint: disable=protected-access
        assert retriever._embeddings[1] == pytest.approx(
            torch.tensor(embeddings.column("embedding").to_pylist()[1]), abs=1e-3
        )


def test_in_memory_retriever(govt_embeddings_file):  # pylint: disable=redefined-outer-name
    """Verify basic functionality of the InMemoryRetriever class"""
    retriever = InMemoryRetriever(govt_embeddings_file, _EMBEDDING_MODEL_NAME)
    result = retriever.retrieve(_EXAMPLE_CHAT_INPUT.messages[-1].content)
    assert result.column("id").to_pylist() == [
        "775449d1aa187ec5",
        "775449d1aa187ec5",
        "775449d1aa187ec5",
        "775449d1aa187ec5",
        "775449d1aa187ec5",
    ]


def test_retrieval_in_memory_request_processor(govt_embeddings_file):  # pylint: disable=redefined-outer-name
    """Basic test of the RequestProcessor with InMemoryRetriever that
    performs RAG retrieval"""
    retriever = InMemoryRetriever(govt_embeddings_file, _EMBEDDING_MODEL_NAME)
    request_processor = RetrievalRequestProcessor(retriever, top_k=3)
    results = request_processor.process(_EXAMPLE_CHAT_INPUT)
    assert len(results) == 1
    # print(results[0].documents)
    assert [d.doc_id for d in results[0].documents] == [
        "775449d1aa187ec5",
        "775449d1aa187ec5",
        "775449d1aa187ec5",
    ]


@pytest.fixture()
def mocked_elasticsearch_retriever():  # pylint: disable=redefined-outer-name
    with mock.patch(
        "elasticsearch.Elasticsearch.search",
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.9,
                        "_source": {"id": "1", "text": "test1"},
                    },
                    {
                        "_id": "2",
                        "_score": 0.8,
                        "_source": {"id": "2", "text": "test2"},
                    },
                ]
            }
        },
    ):
        mocked_retriever = ElasticsearchRetriever(
            corpus_name="test", host="https://localhost:9200"
        )
        yield mocked_retriever


def test_elasticsearch_retriever(mocked_elasticsearch_retriever):  # pylint: disable=redefined-outer-name
    """Verify basic functionality of the ElasticsearchRetriever class"""

    result = mocked_elasticsearch_retriever.retrieve(
        _EXAMPLE_CHAT_INPUT.messages[-1].content, 1
    )
    assert result.column("id").to_pylist() == [
        "1",
        "2",
    ]


def test_retrieval_elasticsearch_request_processor(mocked_elasticsearch_retriever):  # pylint: disable=redefined-outer-name
    """Basic test of the RequestProcessor with ElasticsearchRetriever that
    performs RAG retrieval"""

    request_processor = RetrievalRequestProcessor(
        mocked_elasticsearch_retriever, top_k=3
    )
    results = request_processor.process(_EXAMPLE_CHAT_INPUT)
    # print(results[0].documents)
    assert [d.doc_id for d in results[0].documents] == [
        "1",
        "2",
    ]
