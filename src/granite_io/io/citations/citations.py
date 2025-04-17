# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite citations intrinsic.
"""

# Standard
import json

# Third Party
import nltk

# Local
from granite_io.backend.base import Backend
from granite_io.io.base import (
    InputOutputProcessor,
    ModelDirectInputOutputProcessorWithGenerate,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    ControlsRecord,
    Document,
    Granite3Point2InputProcessor,
    Granite3Point2Inputs,
)
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    Citation,
    GenerateInputs,
    GenerateResults,
)

# The citations intrinsic model expects to see a special system prompt after the last
# assistant message.
_CITATIONS_SYSTEM_PROMPT = (
    "Split the last assistant response into individual "
    "sentences. For each sentence in the response, identify the statement IDs "
    "from the documents that it references. Ensure that your output includes all "
    "response sentence IDs, and for each response sentence ID, provide the "
    "corresponding referring document sentence IDs."
)


# We would like to use a JSON schema for the raw output of the model, but can't due
# to a mismatch between the JSON schema that outlines supports and the current outputs
# of the model.
# Here's the JSON schem we would use:
# _MODEL_OUTPUT_SCHEMA = {
#     "$schema": "https://json-schema.org/draft/2020-12/schema#",
#     "type": "object",
#     "patternProperties": {
#         "^<r[0-9]+>$": {
#             "type": "array",
#             "items": {"type": "string", "pattern": "^<c[0-9]+>$"},
#         }
#     },
#     "additionalProperties": False,
# }

# Instead of JSON schema, we use a regex to constrain the output.
_LIST_ELEMENT_REGEX = r'"<r\d+>": \[(("<c\d+>", )*"<c\d+>")?\]'
_MODEL_OUTPUT_REGEX = (
    r"\{(" + _LIST_ELEMENT_REGEX + r", )*" + _LIST_ELEMENT_REGEX + r"\}"
)


def mark_sentence_boundaries(
    split_strings: list[list[str]], tag_prefix: str
) -> tuple[str, int]:
    """
    Modify one or more input strings by inserting a tag in the form
    ``<[prefix][number]>``
    at the location of each sentence boundary.

    :param split_strings: Input string(s), pre-split into sentences
    :param tag_prefix: String to place before the number part of each tagged
        sentence boundary.

    :returns: List of input strings with all sentence boundaries marked.
    """
    index = 0
    result = []
    for sentences in split_strings:
        to_concat = []
        for sentence in sentences:
            to_concat.append(f"<{tag_prefix}{index}> {sentence}")
            index += 1
        result.append(" ".join(to_concat))
    return result


class CitationsIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the Granite citations intrinsic, also known as the [LoRA Adapter
    for Citation Generation](https://huggingface.co/ibm-granite/
    granite-3.2-8b-lora-rag-citation-generation). 
    
    Takes as input a chat completion and returns a version of the completion with 
    citations to documents.

    Example input to the IO processor's :func`acreate_chat_completion()` call:

    ```
    {
    "messages": [
        {
            "role": "user",
            "content": "What is the visibility level of Git Repos and Issue Tracking \
projects?"
        },
        {
            "role": "assistant",
            "content": "Git Repos and Issue Tracking projects can have one of three \
visibility levels: private, internal, or public. Private projects are visible only to \
project members, internal projects are visible to all users logged in to IBM Cloud, \
and public projects are visible to anyone."
        }
    ],
    "documents": [
        {
            "text": "Git Repos and Issue Tracking is an IBM-hosted component of the \
Continuous Delivery service. All of the data that you provide to Git Repos and Issue \
Tracking, including but not limited to source files, issues, pull requests, and \
project configuration properties, is managed securely within Continuous Delivery. \
However, Git Repos..."
        },
        {
            "text": "After you create a project in Git Repos and Issue Tracking, but \
before you entrust any files, issues, records, or other data with the project, review \
the project settings and change any settings that are necessary to protect your data. \
Settings to review include visibility levels, email notifications, integrations, web \
hooks, access tokens, deploy tokens, and deploy keys...."
        }
    ],
    "generate_inputs": {
        "temperature": 0.0,
        "max_tokens": 1024
    }
}
    ```

    Example prompt that the IO processor would send to the model if it received the
    above input:

    ```
<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: April 09, 2025.
You are Granite, developed by IBM. You are a helpful AI assistant with access to the \
following tools. When a tool is required to answer the user's query, respond with \
<|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the \
provided list of tools, notify the user that you do not have the ability to fulfill \
the request.<|end_of_text|>
<|start_of_role|>documents<|end_of_role|>Document 0
<c0> Git Repos and Issue Tracking is an IBM-hosted component of the Continuous \
Delivery service. <c1> All of the data that you provide to Git Repos and Issue \
Tracking, including but not limited to source files, issues, pull requests, and \
project configuration properties, is managed securely within Continuous Delivery. 
[...]

Document 1
<c25> After you create a project in Git Repos and Issue Tracking, but before you \
entrust any files, issues, records, or other data with the project, review the \
project settings and change any settings that are necessary to protect your data. \
<c26> Settings to review include visibility levels, email notifications, integrations, \
web hooks, access tokens, deploy tokens, and deploy keys. <c27> Project visibility \
[...]

<|start_of_role|>user<|end_of_role|>What is the visibility level of Git Repos and \
Issue Tracking projects?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|><r0> Git Repos and Issue Tracking projects \
can have one of three visibility levels: private, internal, or public. <r1> Private \
projects are visible only to project members, internal projects are visible to all \
users logged in to IBM Cloud, and public projects are visible to anyone.<|end_of_text|>
<|start_of_role|>system<|end_of_role|>Split the last assistant response into \
individual sentences. For each sentence in the response, identify the statement IDs \
from the documents that it references. Ensure that your output includes all response \
sentence IDs, and for each response sentence ID, provide the corresponding referring \
document sentence IDs.<|end_of_text|>
    ```

    Example of raw output of the model for the above request:

    ```
    {"<r0>": ["<c7>"], "<r1>": ["<c8>", "<c10>", "<c11>"]}
    ```
    (note that constrained decoding is required to produce valid JSON reliably)

    Example of processed output from this IO processor for the above raw model output:
    ```
    {
    "content": "Git Repos and Issue Tracking projects can have one of three visibility \
levels: private, internal, or public. Private projects are visible only to \
project members, internal projects are visible to all users logged in to IBM Cloud, \
and public projects are visible to anyone.",
    "role": "assistant",
    "tool_calls": [],
    "reasoning_content": null,
    "citations": [
        {
        "citation_id": "0",
        "doc_id": "0",
        "context_text": "Project visibility levels \n\nGit Repos and Issue Tracking \
projects can have one of the following visibility levels: private, internal, or \
public.",
        "context_begin": 1034,
        "context_end": 1178,
        "response_text": "Git Repos and Issue Tracking projects can have one of three \
visibility levels: private, internal, or public.",
        "response_begin": 0,
        "response_end": 108
        },
        {
        "citation_id": "1",
        "doc_id": "0",
        "context_text": "* Private projects are visible only to project members.",
        "context_begin": 1179,
        "context_end": 1234,
        "response_text": "Private projects are visible only to project members, \
internal projects are visible to all users logged in to IBM Cloud, and public \
projects are visible to anyone.",
        "response_begin": 109,
        "response_end": 272
        },
        {
        "citation_id": "2",
        "doc_id": "0",
        "context_text": "* Internal projects are visible to all users that are \
logged in to IBM Cloud.",
        "context_begin": 1353,
        "context_end": 1430,
        "response_text": "Private projects are visible only to project members, \
internal projects are visible to all users logged in to IBM Cloud, and public \
projects are visible to anyone.",
        "response_begin": 109,
        "response_end": 272
        },
        {
        "citation_id": "3",
        "doc_id": "0",
        "context_text": "* Public projects are visible to anyone.",
        "context_begin": 1431,
        "context_end": 1471,
        "response_text": "Private projects are visible only to project members, \
internal projects are visible to all users logged in to IBM Cloud, and public \
projects are visible to anyone.",
        "response_begin": 109,
        "response_end": 272
        }
    ],
    "documents": null,
    "hallucinations": null,
    "stop_reason": null
    }
    ```
    """

    def __init__(self, backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point2InputProcessor()

        # Object that identifies sentence boundaries. Currently we assume an NLTK
        # sentence tokenizer is used here. This may change in the future.
        # Note that THE SENTENCE SPLITTER MUST BE DETERMINISTIC, because we invoke it
        # twice on every inference request.
        self.sentence_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.messages[-1].role == "assistant":
            raise ValueError("Last message is not an assistant message")
        if inputs.documents is None or len(inputs.documents) == 0:
            raise ValueError("Input does not contain documents")

        # Split the documents into sentences. We will need these sentences later on
        # when parsing the outputs of the model, so save a list of sentences for each
        # document.
        docs_as_sentences = [
            list(self.sentence_splitter.tokenize(d.text)) for d in inputs.documents
        ]

        # The documents input to the model consists of the original documents with
        # each sentence boundary marked with <c0>, <c1>, ... <ck-1>, where `k` is the
        # number of sentences in ALL documents.
        rewritten_docs = [
            Document(text=t) for t in mark_sentence_boundaries(docs_as_sentences, "c")
        ]

        # The last assistant message also needs to be split into sentences. The we
        # encode it with each sentence boundary marked with <r0>, <r1>, ... <rj-1>,
        # where `j` is the number of sentences in the message.
        last_message_as_sentences = list(
            self.sentence_splitter.tokenize(inputs.messages[-1].content)
        )

        rewritten_last_message_text = mark_sentence_boundaries(
            [last_message_as_sentences], "r"
        )[0]
        rewritten_messages = [m.model_copy() for m in inputs.messages]
        rewritten_messages[-1].content = rewritten_last_message_text

        # Put the rewritten docs and last message back into the original chat completion
        # and let the Granite 3.2 IO processor take care of the rest of the formatting.
        rewritten_inputs = inputs.model_copy(
            update={"documents": rewritten_docs, "messages": rewritten_messages}
        )
        prompt = self.base_input_processor.transform(
            rewritten_inputs,
            # No <|start_of_role|>assistant<|end_of_role|> at end of prompt string.
            False,
        )

        # The model's training data uses a special system message to the end of the
        # messages as the prompting string.
        # This is not a valid input for the base model, so we append it to the prompt
        # manually.
        if add_generation_prompt:
            prompt = prompt + (
                f"<|start_of_role|>system<|end_of_role|>{_CITATIONS_SYSTEM_PROMPT}"
                f"<|end_of_text|>"
            )

        generate_inputs_before = (
            inputs.generate_inputs if inputs.generate_inputs else GenerateInputs()
        )

        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Single completion for citations intrinsic
                "n": 1,
                # Always generate citatations at temperature 0
                "temperature": 0.0,
                # Ensure we have enough of a token budget to reliably produce the
                # full output.
                "max_tokens": 1024,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_regex": _MODEL_OUTPUT_REGEX},
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        if inputs is None:
            raise ValueError("Cannot construct results without original inputs")

        # Reconstitute the sentence boundaries so that we can compute offsets
        doc_sentence_offsets = [
            list(self.sentence_splitter.span_tokenize(d.text)) for d in inputs.documents
        ]
        # Flatten into a table for decoding spans
        flat_doc_sentence_offsets = [
            offset for offsets in doc_sentence_offsets for offset in offsets
        ]

        # Do the same for the original response
        message_sentence_offsets = list(
            self.sentence_splitter.span_tokenize(inputs.messages[-1].content)
        )

        # Compute a mapping from sentence number to document number
        sentence_to_doc = []
        for doc_ix, doc_offsets in enumerate(doc_sentence_offsets):
            sentence_to_doc = sentence_to_doc + ([doc_ix] * len(doc_offsets))

        # print(f"{sentence_to_doc=}")

        results = []
        for raw_result in output.results:
            try:
                # Example output:
                # {"<r0>": ["<c6>"], "<r1>": ["<c7>"], "<r2>": ["<c5>"],
                # "<r3>": ["<c6>"], "<r4>": ["<c8>", "<c9>"], "<r5>": ["<c10>"]}
                parsed_json = json.loads(raw_result.completion_string)
                citations = []
                next_citation_id = 0
                content = inputs.messages[-1].content

                if not isinstance(parsed_json, dict):
                    raise TypeError("Model output is not a JSON object")
                for response_key, value in parsed_json.items():
                    # Example: <r0>
                    response_index = int(response_key[2:-1])
                    response_begin, response_end = message_sentence_offsets[
                        response_index
                    ]
                    response_text = content[response_begin:response_end]
                    if not isinstance(value, list):
                        raise TypeError(f"Entry for {response_key} is not a list")
                    for citation_key in value:
                        if not isinstance(citation_key, str):
                            raise TypeError(
                                f"Value in list for {response_key} is not a string"
                            )
                        # Example: "<c10>"
                        citation_index = int(citation_key[2:-1])
                        doc_num = sentence_to_doc[citation_index]
                        context_begin, context_end = flat_doc_sentence_offsets[
                            citation_index
                        ]
                        context_text = inputs.documents[doc_num].text[
                            context_begin:context_end
                        ]

                        # Convert to our current internal format.
                        citations.append(
                            Citation(
                                citation_id=str(next_citation_id),
                                doc_id=str(doc_num),
                                context_text=context_text,
                                context_begin=context_begin,
                                context_end=context_end,
                                response_text=response_text,
                                response_begin=response_begin,
                                response_end=response_end,
                            )
                        )
                        next_citation_id += 1

            except (json.JSONDecodeError, TypeError) as e:
                # This type of error shouldn't happen, because we use constrained
                # decoding.
                content = f"ERROR: {e} (raw output: {raw_result})"
                citations = []

                # TEMPORARY: Pass through errors for now
                raise e

            # print(f"Adding {raw_result.completion_string} as raw result")
            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        content=content,
                        citations=citations,
                        # TEMPORARY -- should be original message's raw result
                        raw=raw_result.completion_string,
                    )
                )
            )

        return ChatCompletionResults(results=results)


class CitationsCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that generates a response, then adds citations.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        lora_backend: Backend,
        request_citations_from_generator: bool = False,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param lora_backend: Backend for running the citations intrinsic.
        :param request_citations_from_generator: if ``True``, invoke ``generator``
         with the Granite ``citations`` control turned on. If ``False``, the value of
         the ``citations`` control will be passed through from requests as is.
        """
        self._generator = generator
        self._citations = CitationsIOProcessor(lora_backend)
        self._request_citations_from_generator = request_citations_from_generator

    def update_request_citations_from_generator(
        self, request_citations_from_generator: bool
    ):
        """Convenience method to update whether to request (and then discard) citations
        output from the generator

        :param request_citations_from_generator: New value to be applied to subsequent
         calls to the I/O processor."""
        self._request_citations_from_generator = request_citations_from_generator

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Downcast to extended Granite inputs. This also creates a copy.
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        if self._request_citations_from_generator:
            # Code above already copied inputs, so we can modify inputs in place
            controls = ControlsRecord() if not inputs.controls else inputs.controls
            controls.citations = True
            inputs.controls = controls

        generator_output = await self._generator.acreate_chat_completion(inputs)

        # Generate citations for all completions in parallel
        futures = []
        for result in generator_output.results:
            futures.append(
                self._citations.acreate_chat_completion(
                    inputs.with_next_message(
                        result.next_message
                    ).with_addl_generate_params(
                        # Only request the top-1 result from the citations intrinsic
                        {"n": 1, "temperature": 0.0}
                    )
                )
            )

        # Wait for citations to come back and bundle them into a result list.
        processed_results = []
        for future in futures:
            citations_output = await future
            processed_results.append(citations_output.results[0])
        return ChatCompletionResults(results=processed_results)
