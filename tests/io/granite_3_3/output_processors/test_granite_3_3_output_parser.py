# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""
Tests for the model output parser
"""

# Standard
import os

# Third Party
from test_utils import load_text_file
import pytest

# Local
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    ControlsRecord,
    Document,
    Granite3Point3Inputs,
)
from granite_io.io.granite_3_3.output_processors.granite_3_3_output_parser import (
    parse_model_output,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


@pytest.fixture(name="inputs", scope="session")
def fixture_inputs():
    # Tests below do not need the inputs, but in the IO flow, an input is
    # necessary, so we create a placeholder input object.
    # Settings on this input object can be correctly enabled, as designed.
    empty = Granite3Point3Inputs.model_validate(
        {"messages": [{"role": "user", "content": ""}]}
    )

    empty_with_citations = empty.model_copy()
    controls_citation = ControlsRecord()
    controls_citation.citations = True
    empty_with_citations.controls = controls_citation

    empty_with_citations_hallucinations = empty.model_copy()
    controls_citation_hallucination = ControlsRecord()
    controls_citation_hallucination.citations = True
    controls_citation_hallucination.hallucinations = True
    empty_with_citations_hallucinations.controls = controls_citation_hallucination

    return {
        "empty": empty,
        "empty_with_citation_control": empty_with_citations,
        "empty_with_citation_hallucination_controls": empty_with_citations_hallucinations,
    }


def test_output(inputs):
    model_output = load_text_file(os.path.join(TEST_DATA_DIR, "test_output.txt"))
    parsed_output = parse_model_output(model_output, inputs["empty"])

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "To efficiently find the fastest way for a seller" in response

    assert parsed_output["docs"] is None
    assert parsed_output["citations"] is None
    assert parsed_output["hallucinations"] is None


def test_output_with_citation(inputs):
    model_output = load_text_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_citation.txt")
    )
    parsed_output = parse_model_output(
        model_output, inputs["empty_with_citation_control"]
    )

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]  # pylint: disable = unsubscriptable-object
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert "NHL in 2019" in doc["text"]

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "most successful hurling team" in response

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
    assert citation["citation_id"] == "0"
    assert citation["doc_id"] == "1"
    assert len(citation["response_text"]) >= 1
    assert citation["response_begin"] == 124
    assert citation["response_end"] == 709
    assert len(citation["context_text"]) >= 1
    assert citation["context_begin"] == 0
    assert citation["context_end"] == 561

    assert parsed_output["hallucinations"] is None


def test_output_with_invalid_citation(inputs):
    model_output = load_text_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_invalid_citation.txt")
    )
    parsed_output = parse_model_output(
        model_output, inputs["empty_with_citation_control"]
    )

    assert parsed_output["citations"] is None
    assert parsed_output["hallucinations"] is None


def test_output_with_colons_in_citation_text(inputs):
    model_output = load_text_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_colons_citation_text.txt")
    )
    parsed_output = parse_model_output(
        model_output, inputs["empty_with_citation_control"]
    )

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]  # pylint: disable = unsubscriptable-object
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert (
        "3부 : [UNK] [UNK] 종료 [UNK], 공영 주택의 세입자는 이 프로그램에서 [UNK] [UNK] 종료에 직면합니다. 이 2가지 상황에 대해 더 자세"
        in doc["text"]
    )

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "Yes, you can visit a law library to conduct your legal research" in response

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
    assert citation["citation_id"] == "0"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1

    hallucinations = parsed_output["hallucinations"]
    assert len(hallucinations) == 4
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
        assert len(hallucination["response_text"]) >= 1
        hallc_id += 1


def test_output_with_citation_hallucinations(inputs):
    model_output = load_text_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_citation_hallucinations.txt")
    )
    parsed_output = parse_model_output(
        model_output, inputs["empty_with_citation_hallucination_controls"]
    )

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]  # pylint: disable = unsubscriptable-object
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
    assert citation["citation_id"] == "0"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1
    assert citation["context_begin"] == 0
    assert citation["context_end"] == 141
    assert len(citation["response_text"]) >= 1
    assert citation["response_begin"] == 47
    assert citation["response_end"] == 443

    hallucinations = parsed_output["hallucinations"]
    assert len(hallucinations) == 2
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


def test_output_with_citation_from_source(inputs):
    model_output = load_text_file(
        os.path.join(TEST_DATA_DIR, "test_output_with_citation_from_source.txt")
    )
    doc_source = load_text_file(os.path.join(TEST_DATA_DIR, "test_document_source.txt"))
    doc_input = [Document(text=f"{doc_source}", doc_id=1)]

    inputs_with_doc = inputs["empty_with_citation_control"]
    inputs_with_doc.documents = doc_input

    parsed_output = parse_model_output(model_output, inputs_with_doc)

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 1
    doc = docs[0]  # pylint: disable = unsubscriptable-object
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert doc["text"] == doc_input[0].text

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert "Limerick" in response

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
    assert citation["citation_id"] == "0"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1
    assert citation["context_begin"] == 0
    assert citation["context_end"] == 561
    assert len(citation["response_text"]) >= 1
    assert citation["response_begin"] == 124
    assert citation["response_end"] == 709

    assert parsed_output["hallucinations"] is None


def test_output_with_multiple_citations_per_document(inputs):
    model_output = load_text_file(
        os.path.join(
            TEST_DATA_DIR, "test_output_with_multiple_citations_per_document.txt"
        )
    )
    parsed_output = parse_model_output(
        model_output, inputs["empty_with_citation_control"]
    )

    assert parsed_output
    assert isinstance(parsed_output, dict)
    keys = ["docs", "response", "citations", "hallucinations"]
    for key in keys:
        assert key in parsed_output

    docs = parsed_output["docs"]
    assert len(docs) == 3
    doc = docs[0]  # pylint: disable = unsubscriptable-object
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "1"
    assert doc["citation_id"] == "0"
    assert "law - libraries / 에서 [UNK] 수 [UNK] 지역 법률 도서관에" in doc["text"]
    doc = docs[1]  # pylint: disable = unsubscriptable-object
    keys = ["doc_id", "text"]
    for key in keys:
        assert key in doc
    assert doc["doc_id"] == "2"
    assert doc["citation_id"] == "1"
    assert (
        "1부 : 머리말 이 시리즈의 비디오에 대한 나머지 부분을 [UNK] 머리말을 들으려면"
        in doc["text"]
    )

    response = parsed_output["response"]
    assert isinstance(response, str)
    assert (
        "Law libraries typically have self-help legal books, official reports, and case laws that could be helpful for your research"
        in response
    )

    citations = parsed_output["citations"]
    assert len(citations) == 3
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
    assert citation["citation_id"] == "0"
    assert citation["doc_id"] == "1"
    assert len(citation["context_text"]) >= 1

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
        assert len(hallucination["response_text"]) >= 1
        hallc_id += 1
