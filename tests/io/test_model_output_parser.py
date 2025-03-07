# SPDX-License-Identifier: Apache-2.0

"""
Tests for the model output parser
"""

# Standard
from pathlib import Path
import os
import re

# Third Party
import pytest

# Local
from granite_io.io.model_output_parser import parse_model_output

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _load_model_output_file(file_name: str) -> str:
    response = Path(file_name).read_text(encoding="UTF-8")
    return response


def test_output():
    model_output = _load_model_output_file(
        os.path.join(TEST_DATA_DIR, "test_output.txt")
    )
    parsed_output = parse_model_output(model_output)

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "To efficiently find the fastest way for a seller" in response

    docs = parsed_output["docs"]
    assert len(docs) == 0

    citations = parsed_output["citations"]
    assert len(citations) == 0

    hallucinations = parsed_output["hallucinations"]
    assert len(hallucinations) == 0


def test_output_with_citation():
    model_output = _load_model_output_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_citation.txt")
    )
    parsed_output = parse_model_output(model_output)

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert "RAG, retrieval-augmented generation" in doc["text"]

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "It's a technique that combines the strengths" in response

    citations = parsed_output["citations"]
    assert len(citations) == 1
    citation = citations[0]
    keys = [
        "citation_id",
        "doc_id",
        "context_text",
        "context_begin",
        "context_end",
        "response_text",
        "response_begin",
        "response_end",
    ]
    for key in keys:
        assert key in citation
    assert citation["citation_id"] == "1"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1
    assert citation["context_begin"] == 0
    assert citation["context_end"] == 141
    assert len(citation["response_text"]) >= 1
    assert citation["response_begin"] == 0
    assert citation["response_end"] == 46

    hallucinations = parsed_output["hallucinations"]
    assert len(hallucinations) == 0


def test_output_with_invalid_citation():
    model_output = _load_model_output_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_invalid_citation.txt")
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Error in extracting citation info: Expected citations but found none"
        ),
    ):
        parse_model_output(model_output)


def test_output_with_citation_hallucinations():
    model_output = _load_model_output_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_citation_hallucinations.txt")
    )
    parsed_output = parse_model_output(model_output)

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert "RAG, retrieval-augmented generation" in doc["text"]

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert len(response) >= 1

    citations = parsed_output["citations"]
    assert len(citations) == 1
    citation = citations[0]
    keys = [
        "citation_id",
        "doc_id",
        "context_text",
        "context_begin",
        "context_end",
        "response_text",
        "response_begin",
        "response_end",
    ]
    for key in keys:
        assert key in citation
    assert citation["citation_id"] == "1"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1
    assert citation["context_begin"] == 0
    assert citation["context_end"] == 141
    assert len(citation["response_text"]) >= 1
    assert citation["response_begin"] == 47
    assert citation["response_end"] == 194

    hallucinations = parsed_output["hallucinations"]
    assert len(hallucinations) == 3
    hallc_id = 1
    for hallucination in hallucinations:
        keys = [
            "hallucination_id",
            "risk",
            "response_text",
            "response_begin",
            "response_end",
        ]
        for key in keys:
            assert key in hallucination
        assert hallucination["hallucination_id"] == str(hallc_id)
        assert hallucination["risk"] == "low"
        assert len(hallucination["response_text"]) >= 1
        hallc_id += 1
