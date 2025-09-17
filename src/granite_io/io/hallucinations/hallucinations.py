# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite hallucinations intrinsic.
"""

# Standard
import json

# Third Party
from pydantic import BaseModel, ConfigDict, NonNegativeInt, RootModel, StrictStr

# Local
from granite_io.backend.base import Backend
from granite_io.io.base import (
    InputOutputProcessor,
    ModelDirectInputOutputProcessorWithGenerate,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    ControlsRecord,
    Granite3Point3InputProcessor,
    Granite3Point3Inputs,
)
from granite_io.optional import nltk_check
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
    Hallucination,
)

# The hallucinations intrinsic model expects to see a special system prompt after the
# last assistant message.
_HALLUCINATION_SYSTEM_PROMPT = (
    "Split the last assistant response into individual sentences. "
    "For each sentence in the last assistant response, identify the faithfulness "
    "by comparing with the provided documents and generate the faithfulness reasoning "
    "and faithfulness decision. "
    "Ensure that your output includes all response sentence IDs, "
    "and for each response sentence ID, provide the corresponding faithfulness "
    "reasoning and faithfulness decision. "
    "The output must be a json structure."
)


# Specify the schema of the raw output of the model to use in contrained decoding
# This is done by first creating a Pydantic model representing the raw model output
# and converting it to a JSON schema
class _MODEL_OUTPUT_ENTRY(BaseModel):
    i: NonNegativeInt
    r: StrictStr
    f: StrictStr

    model_config = ConfigDict(extra="forbid")


class _MODEL_OUTPUT(RootModel):
    root: list[_MODEL_OUTPUT_ENTRY]


_MODEL_OUTPUT_SCHEMA = _MODEL_OUTPUT.model_json_schema()


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


class HallucinationsIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the Granite hallucinations intrinsic, also known as the [LoRA
    Adapter for Hallucination Detection](https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib/blob/main/hallucination_detection_lora/README.md).

    Takes as input a chat completion and returns a version of the completion with
    hallucinations detected on the last assistant turn.

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
Today's Date: June 16, 2025.
You are Granite, developed by IBM. Write the response to the user\'s input \
by strictly aligning with the facts in the provided documents. If the \
information needed to answer the question is not available in the documents, \
inform the user that the question cannot be answered based on the available data.\
<|end_of_text|>
<|start_of_role|>document {"document_id": "1"}<|end_of_role|>
Git Repos and Issue Tracking is an IBM-hosted component of the Continuous Delivery \
service. All of the data that you provide to Git Repos and Issue Tracking, \
including but not limited to source files, issues, [...] <|end_of_text|>
<|start_of_role|>document {"document_id": "2"}<|end_of_role|>
After you create a project in Git Repos and Issue Tracking, but before you \
entrust any files, issues, records, or other data with the project, review \
the project settings and change any settings that are necessary to protect \
your data. Settings to review include visibility [...] <|end_of_text|>
<|start_of_role|>user<|end_of_role|>What is the visibility level of Git Repos and \
Issue Tracking projects?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|><i0> Git Repos and Issue Tracking projects \
can have one of three visibility levels: private, internal, or public. <i1> Private \
projects are visible only to project members, internal projects are visible to all \
logged-in IBM Cloud users, and public projects are visible to anyone.<|end_of_text|>
<|start_of_role|>system<|end_of_role|>Split the last assistant response into individual\
sentences. For each sentence in the last assistant response, identify the faithfulness \
by comparing with the provided documents and generate the faithfulness reasoning and \
faithfulness decision. Ensure that your output includes all response sentence IDs, and \
for each response sentence ID, provide the corresponding faithfulness reasoning and \
faithfulness decision. The output must be a json structure.<|end_of_text|>
    ```

    Example of raw output of the model for the above request:

    ```
    "[{\"i\": 0, \"r\": \"This sentence makes a factual claim about the visibility \
levels of Git Repos and Issue Tracking projects. The document states 'Git Repos and \
Issue Tracking projects can have one of the following visibility levels: private, \
internal, or public.' This matches exactly with the claim in the sentence. \", \"f\": \
\"faithful\"}, ...]"
    ```
    Note that the raw model output is JSON data encoded as a JSON string.

    Example of processed output from this IO processor for the above raw model output:
    ```
    {
        "content": "Git Repos and Issue Tracking projects can have one of three \
visibility levels: private, internal, or public. Private projects are visible only to \
project members, internal projects are visible to all logged-in IBM Cloud users, and \
public projects are visible to anyone.",
        "role": "assistant",
        "tool_calls": [],
        "reasoning_content": null,
        "citations": null,
        "documents": null,
        "hallucinations": [
            {
            "hallucination_id": "0",
            "risk": "faithful",
            "reasoning": "This sentence makes a factual claim about the visibility \
levels of Git Repos and Issue Tracking projects. The document states 'Git Repos and \
Issue Tracking projects can have one of the following visibility levels: private, \
internal, or public.' This matches exactly with the claim in the sentence.",
            "response_text": "Git Repos and Issue Tracking projects can have one of \
three visibility levels: private, internal, or public.",
            "response_begin": 0,
            "response_end": 108
            },
            {
            "hallucination_id": "1",
            "risk": "faithful",
            "reasoning": "This sentence makes factual claims about the visibility of \
each type of project. The document states 'Private projects are visible only to \
project members,' 'Internal projects are visible to all users that are logged in to \
IBM Cloud,' and 'Public projects are visible to anyone.' These statements match \
exactly with the claims in the sentence.",
            "response_text": "Private projects are visible only to project members, \
internal projects are visible to all logged-in IBM Cloud users, and public projects \
are visible to anyone.",
            "response_begin": 109,
            "response_end": 269
            }
        ],
        "stop_reason": "stop"
    }
    ```
    """

    def __init__(self, backend):
        super().__init__(backend=backend)
        with nltk_check("LoRA Adapter for Hallucination Detection in RAG outputs"):
            # Third Party
            import nltk

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point3InputProcessor()

        # Object that identifies sentence boundaries. Currently we assume an NLTK
        # sentence tokenizer is used here. This may change in the future.
        # Note that THE SENTENCE SPLITTER MUST BE DETERMINISTIC, because we invoke it
        # twice on every inference request.
        self.sentence_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.messages[-1].role == "assistant":
            raise ValueError("Last message is not an assistant message")
        if inputs.documents is None or len(inputs.documents) == 0:
            raise ValueError("Input does not contain documents")

        # The last assistant message also needs to be split into sentences. The we
        # encode it with each sentence boundary marked with <r0>, <r1>, ... <rj-1>,
        # where `j` is the number of sentences in the message.
        last_message_as_sentences = list(
            self.sentence_splitter.tokenize(inputs.messages[-1].content)
        )

        rewritten_last_message_text = mark_sentence_boundaries(
            [last_message_as_sentences], "i"
        )[0]
        rewritten_messages = [m.model_copy() for m in inputs.messages]
        rewritten_messages[-1].content = rewritten_last_message_text

        # Put the rewritten docs and last message back into the original chat completion
        # and let the Granite 3.3 IO processor take care of the rest of the formatting.
        rewritten_inputs = inputs.model_copy(
            update={"documents": inputs.documents, "messages": rewritten_messages}
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
                f"<|start_of_role|>system<|end_of_role|>{_HALLUCINATION_SYSTEM_PROMPT}"
                f"<|end_of_text|>"
            )

        generate_inputs_before = (
            inputs.generate_inputs if inputs.generate_inputs else GenerateInputs()
        )

        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Single completion for hallucinations intrinsic
                "n": 1,
                # Always generate hallucinations at temperature 0
                "temperature": 0.0,
                # Ensure we have enough of a token budget to reliably produce the
                # full output.
                "max_tokens": 4096,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_json": _MODEL_OUTPUT_SCHEMA},
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        if inputs is None:
            raise ValueError("Cannot construct results without original inputs")

        # Do the same for the original response
        message_sentence_offsets = list(
            self.sentence_splitter.span_tokenize(inputs.messages[-1].content)
        )

        results = []
        for raw_result in output.results:
            try:
                # Example output:
                # [{"i": 0, "r": "....", "f": "..."}, {"i": 1, "r": "....", "f":
                # "..."},...]
                parsed_json = json.loads(raw_result.completion_string)
                hallucinations = []
                next_hallucination_id = 0
                content = inputs.messages[-1].content

                if not isinstance(parsed_json, list):
                    raise TypeError("Model output is not a JSON array")
                for entry in parsed_json:
                    response_index = entry["i"]
                    if not isinstance(response_index, int):
                        raise TypeError(f"{response_index} is not an integer")
                    if response_index >= len(message_sentence_offsets):
                        # Hallucinated sentence offset
                        print(
                            f"Warning: Skipping out-of-range sentence offset "
                            f"{response_index}"
                        )
                        continue
                    response_begin, response_end = message_sentence_offsets[
                        response_index
                    ]
                    response_text = content[response_begin:response_end]
                    reasoning = entry["r"]
                    value = entry["f"]
                    if not isinstance(value, str):
                        raise TypeError(f"Entry for {response_index} is not a str")
                    if not isinstance(reasoning, str):
                        raise TypeError(f"Entry for {response_index} is not a str")
                    hallucinations.append(
                        Hallucination(
                            hallucination_id=str(next_hallucination_id),
                            response_text=response_text,
                            risk=value,
                            reasoning=reasoning,
                            response_begin=response_begin,
                            response_end=response_end,
                        )
                    )
                    next_hallucination_id += 1  # noqa: SIM113

            except (json.JSONDecodeError, TypeError) as e:
                # This type of error shouldn't happen, because we use constrained
                # decoding.
                content = f"ERROR: {e} (raw output: {raw_result})"
                hallucinations = []

                # TEMPORARY: Pass through errors for now
                raise e

            # print(f"Adding {raw_result.completion_string} as raw result")
            next_message = inputs.messages[-1].model_copy(
                update={
                    "content": content,
                    "hallucinations": hallucinations,
                    # TEMPORARY -- should be original message's raw result
                    "_raw": raw_result.completion_string,
                }
            )

            results.append(ChatCompletionResult(next_message=next_message))

        return ChatCompletionResults(results=results)


class HallucinationsCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that generates a response, then adds hallucinations.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        lora_backend: Backend,
        request_hallucinations_from_generator: bool = False,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param lora_backend: Backend for running the hallucinations intrinsic.
        :param request_hallucinations_from_generator: if ``True``, invoke ``generator``
         with the Granite ``hallucinations`` control turned on. If ``False``, the
         value of the ``hallucinations`` control will be passed through from requests
         as is.
        """
        self._generator = generator
        self._hallucinations = HallucinationsIOProcessor(lora_backend)
        self._request_hallucinations_from_generator = (
            request_hallucinations_from_generator
        )

    def update_request_hallucinations_from_generator(
        self, request_hallucinations_from_generator: bool
    ):
        """Convenience method to update whether to request (and then discard)
        hallucinations output from the generator

        :param request_hallucinations_from_generator: New value to be applied to
         subsequent calls to the I/O processor."""
        self._request_hallucinations_from_generator = (
            request_hallucinations_from_generator
        )

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Downcast to extended Granite inputs. This also creates a copy.
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        if self._request_hallucinations_from_generator:
            # Code above already copied inputs, so we can modify inputs in place
            controls = ControlsRecord() if not inputs.controls else inputs.controls
            controls.hallucinations = True
            inputs.controls = controls

        generator_output = await self._generator.acreate_chat_completion(inputs)

        # Generate hallucinations for all completions in parallel
        futures = []
        for result in generator_output.results:
            futures.append(
                self._hallucinations.acreate_chat_completion(
                    inputs.with_next_message(
                        result.next_message
                    ).with_addl_generate_params(
                        # Only request the top-1 result from the LoRA
                        {"n": 1, "temperature": 0.0}
                    )
                )
            )

        # Wait for hallucinations to come back and bundle them into a result list.
        processed_results = []
        for future in futures:
            hallucinations_output = await future
            processed_results.append(hallucinations_output.results[0])
        return ChatCompletionResults(results=processed_results)
