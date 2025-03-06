# Standard
import copy
import logging
import re

# Third Party
from nltk import sent_tokenize  # pylint: disable=import-error


def _find_substring_in_text(substring: str, text: str) -> list[int]:
    """
    Given two strings - substring and text - find and return all
    matches of substring within text. For each match return its begin and end index
    """
    span_matches = []

    matches_iter = re.finditer(re.escape(substring), text)
    for match in matches_iter:
        span_matches.append({"begin_idx": match.start(), "end_idx": match.end()})

    return span_matches


def _parse_hallucinations_text(hallucinations_text: str) -> list[dict]:
    """
    Given the hallucinations text output by model under the "# Hallucinations:"
    section, extract the hallucinations info as an array of the form:

    [
        {
            "hallucination_id": "Hallucination ID output by model",
            "risk": "Hallucination risk flag",
            "response_text": "Substring of response text for which
                                hallucination risk is computed"
        },
        ...
    ]
    """

    hallucinations = []

    # Find begin spans of all hallucinations
    matches_iter = re.finditer(
        "(\d+)\.\s*Risk (low|high|unanswerable):",  # pylint: disable=anomalous-backslash-in-string # noqa: E501
        hallucinations_text,
    )
    matches = []
    for match in matches_iter:
        matches.append({"match_begin": match.start()})

    assert len(matches) > 0, """Error in extracting hallucination info: \
    Expected hallucination info but found none"""

    # For each hallucination, extract its components (hallucination ID,
    # risk, response text)
    for i in range(len(matches)):  # pylint: disable=consider-using-enumerate
        cur_match = matches[i]

        # Select text corresponding to hallucination (which is the text from the
        # beginning of the hallucination until the beginning of the next
        # hallucination or the end of the text; whichever comes first)
        if i + 1 < len(matches):
            next_match_begin = matches[i + 1]["match_begin"] - 1
        else:
            next_match_begin = len(hallucinations_text)
        hallucination_str = hallucinations_text[
            cur_match["match_begin"] : next_match_begin
        ]

        # Within the hallucination text, extract the citation components
        # (hallucination ID, risk, response text)
        # Use ?s flag to include newlines in match
        matches_iter = re.finditer(
            "(?s)(\d+)\.\s*Risk (low|high|unanswerable): (.+)$",  # pylint: disable=anomalous-backslash-in-string
            hallucination_str,
        )
        idx = 0
        for match in matches_iter:
            cur_hallucination = {
                "hallucination_id": match.group(1),
                "risk": match.group(2),
                "response_text": match.group(3),
            }
            hallucinations.append(cur_hallucination)

            idx += 1

        if idx == 0:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in finding components of hallucination: \
                            Expected single RegEx match but found none.""")
        if idx > 1:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in finding components of hallucination: \
                            Expected single RegEx match but found several.""")

    return hallucinations


def _add_hallucination_response_spans(
    hallucination_info: list[dict], response_text_without_citations: str
) -> list[dict]:
    """
    Given the response text (cleaned from citation tags) and a
    parsed hallucinations info of the form:

    [
        {
            "hallucination_id": "Hallucination ID output by model",
            "risk": "Hallucination risk flag",
            "response_text": "Substring of response text for which hallucination
                                risk is computed"
        },
        ...
    ]

    add to each hallucination element in the array the following attributes
    (the "response_text" replaces the attribute of the same name):

        "response_text": "The response text corresponding to the hallucination
                            element cleaned from ciation tags"
        "response_begin": "The begin index of "response_text" within the response
                            text (without citation tags)"
        "response_end": "The end index of "response_text" within the response
                            text (without citation tags)"
    """

    augmented_hallucination_info = copy.deepcopy(hallucination_info)

    for hallucination in augmented_hallucination_info:
        hallucination_response_text_without_citations = (
            _remove_citations_from_response_text(hallucination["response_text"])
        )
        matches = _find_substring_in_text(
            hallucination_response_text_without_citations,
            response_text_without_citations,
        )
        if len(matches) == 0:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in adding the response spans to hallucination: \
                            Hallucination text not found in response""")

        if len(matches) > 1:
            logging.warning("""Hallucination text found multiple times in \
                            response: Selecting first match""")
        hallucination["response_text"] = hallucination_response_text_without_citations
        hallucination["response_begin"] = matches[0]["begin_idx"]
        hallucination["response_end"] = matches[0]["end_idx"]

    return augmented_hallucination_info


def _parse_citations_text(citations_text: str) -> list[dict]:
    """
    Given the citations text output by model under the "# Citations:" section,
    extract the citation info as an array of the form:

    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context"
        },
        ...
    ]
    """

    citations = []

    # Find begin spans of all citations
    matches_iter = re.finditer("<co>(\d+)</co>", citations_text)  # pylint: disable=anomalous-backslash-in-string
    matches = []
    for match in matches_iter:
        matches.append({"match_begin": match.start()})

    assert len(matches) > 0, (
        "Error in extracting citation info: Expected citations but found none"
    )

    # For each citation, extract its components (citation ID, doc ID, context text)
    for i in range(len(matches)):  # pylint: disable=consider-using-enumerate
        cur_match = matches[i]

        # Select text corresponding to citation (which is the text from the beginning
        # of the citation until the beginning of the next citation or the end of the
        # text; whichever comes first)
        if i + 1 < len(matches):
            next_match_begin = matches[i + 1]["match_begin"] - 1
        else:
            next_match_begin = len(citations_text)
        citation_str = citations_text[cur_match["match_begin"] : next_match_begin]

        # Within the citation text, extract the citation components
        # (citation ID, doc ID, context text)
        # Use ?s flag to include newlines in match
        matches_iter = re.finditer(
            '(?s)<co>(\d+)</co>\s*Document (\d+): "(.+)$',  # pylint: disable=anomalous-backslash-in-string # noqa: E501
            citation_str,
        )
        idx = 0
        for match in matches_iter:
            # If the last character is a double quote (as expected), do not include
            # it in the string
            if match.group(3)[-1] == '"':
                context_text = match.group(3)[:-1]
            # Otherwise, continue but show a warning that there is an error in the
            # output format
            else:
                context_text = match.group(3)
                logging.warning(f"""Last character of citation is not a double \
                                quote: {context_text}""")
            cur_citation = {
                "citation_id": match.group(1),
                "doc_id": match.group(2),
                "context_text": context_text,
            }
            citations.append(cur_citation)

            # If the citation contains a nested Document x mention, then show a warning
            if re.search(r"\nDocument (\d+)", cur_citation["context_text"]):
                logging.warning(f"""Citation text contains another document mention: \
                                {cur_citation["context_text"]}""")

            idx += 1

        if idx == 0:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in finding components of citation: \
                            Expected single RegEx match but found none.""")
        if idx > 1:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in finding components of citation: \
                            Expected single RegEx match but found several.""")

    return citations


def _add_citation_context_spans(
    citation_info: list[dict], docs: list[dict]
) -> list[dict]:
    """
    Given a set of docs and an array of citations of the form:

    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context"
        },
        ...
    ]

    add to each citation in the array the following two attributes:

        "context_begin": "The begin index of "context_text" within document with
                            ID doc_id"
        "context_end": "The end index of "context_text" within document with ID doc_id"
    """
    augmented_citation_info = copy.deepcopy(citation_info)
    docs_by_docid = _create_dict(docs, "doc_id")
    for citation in augmented_citation_info:
        matches = _find_substring_in_text(
            citation["context_text"], docs_by_docid[citation["doc_id"]]["text"]
        )
        if len(matches) == 0:
            # pylint: disable=broad-exception-raised
            raise Exception("""Error in adding the context spans to citation: \
                            Cited text not found in corresponding document""")

        if len(matches) > 1:
            logging.warning("""Cited text found multiple times in corresponding \
                            document: Selecting first match""")
        citation["context_begin"] = matches[0]["begin_idx"]
        citation["context_end"] = matches[0]["end_idx"]

    return augmented_citation_info


def _add_citation_response_spans(
    citation_info: list[dict],
    response_text_with_citations: str,
    response_text_without_citations: str,
) -> list[dict]:
    """
    Given the response text in two forms (the original with citation tags and
    the processed without) and an array of parsed citations of the form:

    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context",
            "context_begin": "The begin index of "context_text" within document with
                                ID doc_id" (Optional)
            "context_end": "The end index of "context_text" within document with
                                ID doc_id" (Optional)
        },
        ...
    ]

    add to each citation in the array the following two attributes:
        "response_text": "The substring of the response for which the citation is
                            provided"
        "response_begin": "The begin index of "response_text" within the response text"
        "response_end": "The end index of "response_text" within the response text"
    """

    augmented_citation_info = copy.deepcopy(citation_info)

    # Split response into sentences
    response_sentences = sent_tokenize(response_text_with_citations)

    # Create dictionary of the response sentence (cleaned from citations) corresponding
    # to each citation ID
    response_sents_by_citation_id = {}
    for sent_idx, sent in enumerate(response_sentences):
        matches_iter = re.finditer("<co>(\d+)</co>", sent)  # pylint: disable=anomalous-backslash-in-string
        for match in matches_iter:
            citation_id = match.group(1)
            if citation_id not in response_sents_by_citation_id:
                sent_without_citations = _remove_citations_from_response_text(sent)
                # Fixes for sentence splitting issue: Associate citation with previous
                # sentence if either of the following holds:
                # - The citation ID appears at the beginning of a sentence
                # - The found sentence is empty after removing citations
                if (match.start() == 0) or (len(sent_without_citations) == 0):
                    if sent_idx > 0:
                        sent_without_citations = _remove_citations_from_response_text(
                            response_sentences[sent_idx - 1]
                        )
                    else:
                        raise ValueError("""Error in extracting the response sentence \
                                        of a citation: Found empty sentence""")
                response_sents_by_citation_id[citation_id] = sent_without_citations
            else:
                raise ValueError("""Error in extracting the response sentence of a \
                                citation: Citation ID appears in more than one \
                                response sentences""")

    # For each citation bring the response sentence to which it refers and its
    # begin/end spans
    for citation in augmented_citation_info:
        if citation["citation_id"] in response_sents_by_citation_id:
            response_text = response_sents_by_citation_id[citation["citation_id"]]
            matches = _find_substring_in_text(
                response_text, response_text_without_citations
            )
            if len(matches) == 0:
                # pylint: disable=broad-exception-raised
                raise Exception("""Error in extracting the response sentence of a \
                                citation: Unexpected error""")

            if len(matches) > 1:
                # Find the citation ID and the text preceding it
                citation_id_matches_iter = re.finditer(
                    "<co>" + citation["citation_id"] + "</co>",
                    response_text_with_citations,
                )
                citation_id_matches = tuple(citation_id_matches_iter)
                assert (
                    len(citation_id_matches) > 0
                ), """Error in extracting the response sentence of a citation: \
                    Citation ID does not appear in the response text"""
                citation_id_match_begin = citation_id_matches[0].start()

                text_before_citation_id = response_text_with_citations[
                    :citation_id_match_begin
                ]
                text_before_citation_id_without_citations = (
                    _remove_citations_from_response_text(text_before_citation_id)
                )

                # Find citation response text as the last match in the response text
                # preceding the citation ID. To do that, first remove the punctuation
                # from the end of the response text, since that would appear after the
                # citation ID
                search_str = re.sub(r"[\.\?\:\;]+$", "", response_text)
                response_text_matches = _find_substring_in_text(
                    search_str, text_before_citation_id_without_citations
                )
                assert len(citation_id_matches) > 0, (
                    "Error in extracting the response sentence of a citation"
                )
                last_response_text_match = response_text_matches[-1]

                citation["response_text"] = response_text
                citation["response_begin"] = last_response_text_match["begin_idx"]
                citation["response_end"] = last_response_text_match["begin_idx"] + len(
                    response_text
                )

                assert (
                    citation["response_text"]
                    == response_text_without_citations[
                        citation["response_begin"] : citation["response_end"]
                    ]
                ), "Error in extracting the response sentence of a citation"
            else:
                citation["response_text"] = response_text
                citation["response_begin"] = matches[0]["begin_idx"]
                citation["response_end"] = matches[0]["end_idx"]
        else:
            raise ValueError("""Error in extracting the response sentence of a \
                            citation: Citation ID does not appear in the response \
                            text""")

    return augmented_citation_info


def _convert_doc_strs_to_dicts(docs: str) -> list[dict]:
    """
    Given a multi-line string with document information per line, extract
    and add to dictionary list with "doc_id" and "text" fields
    """
    doc_dicts = []
    start_citation = "<co>"
    end_citation = "</co>"
    colon = ":"
    if not docs or docs.isspace():
        return doc_dicts
    for line in docs.splitlines():
        if not line or line.isspace():
            continue
        if start_citation not in line or end_citation not in line or colon not in line:
            continue
        doc_id = line[
            line.find(start_citation) + len(start_citation) : line.rfind(end_citation)
        ].strip()
        line_separated = line.split(colon, 1)
        if len(line_separated) <= 1:
            continue
        text = line_separated[1].strip().strip('"')
        doc_dicts.append({"doc_id": doc_id, "text": text})
    return doc_dicts


def _create_dict(input_array: object, key_attrib_name: str) -> dict:
    """
    Given an array and the name of an attribute within the array, return a dictionary
    containing the contents of the array indexed by the given attribute
    """
    new_dict = {}

    for item in input_array:
        if item[key_attrib_name] in new_dict:
            # pylint: disable=broad-exception-raised
            raise Exception(
                """Found unexpected duplicate key while creating \
                            dictionary: """
                + item[key_attrib_name]
            )

        new_dict[item[key_attrib_name]] = item

    return new_dict


def _remove_citations_from_response_text(response_text: str) -> str:
    """
    Given a response text (potentially containing inline <co>...</co> tags),
    return the response text cleaned up from the <co>...</co> tags
    """
    return re.sub(r"\s*<co>\d+</co>\s*", "", response_text)


def _validate_response(response_text: str, citation_info: object):
    if re.search(
        r"<co>(?:(?!(<co>|</co>)).)*<co>(?:(?!(<co>|</co>)).)*</co>", response_text
    ):
        logging.warning(f"Response contains nested <co> tags: {response_text}")

    opening_tag_count = response_text.count("<co>")
    closing_tag_count = response_text.count("</co>")

    if opening_tag_count != closing_tag_count:
        logging.warning(f"""Response contains different number of <co> and </co> \
                        tags: {response_text}""")

    if opening_tag_count != len(citation_info):
        logging.warning(f"""Response contains different number of citations than those \
                        mentioned under '# Citations': {response_text}""")


def _split_model_output_into_parts(model_output: str) -> tuple[str, str, str]:
    """
    Divide the model output into its constituent parts: response_text, citations_text,
    and hallucinations_text.

    This assumes that the model output is of the following form:

    response_text
    # Citations
    citations_text
    # Hallucinations
    hallucinations_text

    where:
    - "# Citations" and "# Hallucinations> are literals, and
    - response_text, citations_text, hallucinations_text are variables corresponding
        the parts of the output

    Note:
    - The citations and hallucinations sections are both optional
    """
    response_text = ""
    citations_text = ""
    hallucinations_text = ""

    if "# Hallucinations:" in model_output:
        response_citations_text, hallucinations_text = model_output.split(
            "# Hallucinations:"
        )
    else:
        response_citations_text = model_output

    if "# Citations:" in model_output:
        response_text, citations_text = response_citations_text.split("# Citations:")
    else:
        response_text = response_citations_text

    return response_text.strip(), citations_text.strip(), hallucinations_text.strip()


def _validate_spans_in_parser_output(parsed_task: object):
    """
    Validate that the hallunication and citation spans correspond to
    the model reponse
    """
    for hallucination in parsed_task["hallucinations"]:
        assert (
            hallucination["response_text"]
            == parsed_task["response"][
                hallucination["response_begin"] : hallucination["response_end"]
            ]
        ), "Unexpected error in generated hallucination response span"
    for citation in parsed_task["citations"]:
        assert (
            citation["response_text"]
            == parsed_task["response"][
                citation["response_begin"] : citation["response_end"]
            ]
        ), "Unexpected error in generated citation response span"
        docs_by_id = _create_dict(parsed_task["docs"], "doc_id")
        assert (
            citation["context_text"]
            == docs_by_id[citation["doc_id"]]["text"][
                citation["context_begin"] : citation["context_end"]
            ]
        ), "Unexpected error in generated citation context span"


def parse_model_output(model_output: str) -> dict[str, dict]:
    """
    Parse the constituents of the output (response) of a model into
    a format where they can be accessed individually

    Args:
        model_output: The response from model request
    Returns:
        Parsed part of the model output as follows:
            "docs": Document references
            "response": Model reponse without citations,
            "citations": Citations,
            "hallucinations": Hallucinations
    }
    """

    doc_dicts = {}
    citations = model_output.split("# Citations:")
    if len(citations) == 2:
        docs = citations[1]
        # Covert docs str to dictionary for parsing
        doc_dicts = _convert_doc_strs_to_dicts(docs)
    else:
        raise RuntimeError(f"""Unable to separate reponse by citation:
                        {citations}""")

    # Split model output into its parts: response, citation, and hallucination section
    response_text, citations_text, hallucinations_text = _split_model_output_into_parts(
        model_output
    )

    # Model output
    logging.info(f"Model output:\n\n{model_output}\n")

    # Parsed response text
    response_text_without_citations = _remove_citations_from_response_text(
        response_text
    ).strip()
    logging.info(f"Parsed response text:\n\n{response_text_without_citations}\n")

    # Parse hallucinations text
    if len(hallucinations_text) > 0:
        hallucination_info = _parse_hallucinations_text(hallucinations_text)
        augmented_hallucination_info = _add_hallucination_response_spans(
            hallucination_info, response_text_without_citations
        )
    else:
        augmented_hallucination_info = []
    logging.info(f"Parsed hallucination info:\n\n{augmented_hallucination_info}\n")

    # Parse citations text
    if len(citations_text) > 0:
        citation_info = _parse_citations_text(citations_text)
        citation_info_with_context_spans = _add_citation_context_spans(
            citation_info, doc_dicts
        )
        citation_info_with_context_response_spans = _add_citation_response_spans(
            citation_info_with_context_spans,
            response_text,
            response_text_without_citations,
        )
        _validate_response(response_text, citation_info)
    else:
        citation_info_with_context_response_spans = []
    logging.info(f"""Parsed citation info:\n\n\
                 {citation_info_with_context_response_spans}\n""")

    # Join all objects into single output
    result = {
        "docs": doc_dicts,
        "response": response_text_without_citations,
        "citations": citation_info_with_context_response_spans,
        "hallucinations": augmented_hallucination_info,
    }
    logging.info(f"Combined parser output:\n\n{result}\n")

    # Validate spans in parser output by checking if the citation/response text
    # matches the begin/end spans
    _validate_spans_in_parser_output(result)

    return result
