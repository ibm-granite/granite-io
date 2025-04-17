# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite hallucinations intrinsic.
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
    RequestProcessor,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    ControlsRecord,
    Granite3Point2InputProcessor,
    Granite3Point2Inputs,
)
from granite_io.types import (
    AssistantMessage,
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
    "score range. "
    "Ensure that your output includes all response sentence IDs, and for each response "
    "sentence ID, provide the corresponding faithfulness score range. "
    "The output must be a json structure."
)

_FAITHFULNESS_RISK_CONVERT = {
    "0.1-0": "0.9-1.0",
    "0.1-0.2": "0.8-0.9",
    "0.2-0.3": "0.7-0.8",
    "0.3-0.4": "0.6-0.7",
    "0.4-0.5": "0.5-0.6",
    "0.5-0.6": "0.4-0.5",
    "0.6-0.7": "0.3-0.4",
    "0.7-0.8": "0.2-0.3",
    "0.8-0.9": "0.1-0.2",
    "0.9-1.0": "0-0.1",
    "unanswerable": "unanswerable",
    "NA": "NA",
}


def faithfulness_to_risk(input_string):
    """
    Takes an input string returns the corresponding
    output string from the dictionary. Uses try-except for error handling.
    """
    try:
        return _FAITHFULNESS_RISK_CONVERT[input_string]
    except KeyError:
        # Handle the case where the key is not found
        return "NA"  # Or raise an exception, or return a default value


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
    I/O processor for the Granite hallucinations intrinsic, also known as the 
    [LoRA Adapter for Hallucination Detection in RAG outputs](
        https://huggingface.co/ibm-granite/
        granite-3.2-8b-lora-rag-hallucination-detection
    )

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
Git Repos and Issue Tracking is an IBM-hosted component of the Continuous \
Delivery service. All of the data that you provide to Git Repos and Issue \
Tracking, including but not limited to source files, issues, pull requests, and \
project configuration properties, is managed securely within Continuous Delivery. 
[...]

Document 1
After you create a project in Git Repos and Issue Tracking, but before you \
entrust any files, issues, records, or other data with the project, review the \
project settings and change any settings that are necessary to protect your data. \
Settings to review include visibility levels, email notifications, integrations, \
web hooks, access tokens, deploy tokens, and deploy keys. Project visibility \
[...]

<|start_of_role|>user<|end_of_role|>What is the visibility level of Git Repos and \
Issue Tracking projects?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|><r0> Git Repos and Issue Tracking projects \
can have one of three visibility levels: private, internal, or public. <r1> Private \
projects are visible only to project members, internal projects are visible to all \
users logged in to IBM Cloud, and public projects are visible to anyone.\
<|end_of_text|> \
<|start_of_role|>system<|end_of_role|>"Split the last assistant response into \
individual sentences. \
For each sentence in the last assistant response, identify the faithfulness score \
range. \
Ensure that your output includes all response sentence IDs, and for each response \
sentence ID, \
provide the corresponding faithfulness score range. \
The output must be a json structure.<|end_of_text|>
    ```

    Example of raw output of the model for the above request:

    ```
    "{\\"<r0>\\": \\"0.2-0.3\\", \\"<r1>\\": \\"0.9-1.0\\"}"
    ```
    Note that the raw model output is JSON data encoded as a JSON string.

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
    "hallucinations": [
        {
          "hallucination_id": "0",
          "risk": "0.1-0.2",
          "response_text": "Git Repos and Issue Tracking projects can have one of \
three visibility levels: private, internal, or public.",
          "response_begin": 0,
          "response_end": 108
        },
        {
          "hallucination_id": "1",
          "risk": "0.9-1.0",
          "response_text": "Private projects are only visible to project members, \
internal projects are visible to all users logged in to IBM Cloud, and public projects \
are visible to anyone.",
          "response_begin": 109,
          "response_end": 272
        }
    ],
    "hallucinations": null,
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
                # Always generate citatations at temperature 0
                "temperature": 0.0,
                # Ensure we have enough of a token budget to reliably produce the
                # full output.
                "max_tokens": 1024,
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
                # Output of the model is JSON packed into a quoted string for some
                # reason. So parse twice.
                parsed_json = json.loads(raw_result.completion_string)
                parsed_json = json.loads(parsed_json)
                hallucinations = []
                next_hallucination_id = 0
                content = inputs.messages[-1].content

                if not isinstance(parsed_json, dict):
                    raise TypeError(
                        f"Model output '{raw_result.completion_string}' "
                        f"is not a JSON object"
                    )
                for response_key, value in parsed_json.items():
                    # Example: <r0>
                    response_index = int(response_key[2:-1])
                    response_begin, response_end = message_sentence_offsets[
                        response_index
                    ]
                    response_text = content[response_begin:response_end]
                    if not isinstance(value, str):
                        raise TypeError(f"Entry for {response_key} is not a str")
                    hallucinations.append(
                        Hallucination(
                            hallucination_id=str(next_hallucination_id),
                            response_text=response_text,
                            risk=faithfulness_to_risk(value),
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
            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        content=content,
                        hallucinations=hallucinations,
                        # TEMPORARY -- should be original message's raw result
                        raw=raw_result.completion_string,
                    )
                )
            )

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
        self.request_hallucinations_from_generator = (
            request_hallucinations_from_generator
        )

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Downcast to extended Granite inputs. This also creates a copy.
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

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


class HallucinationsRequestProcessor(RequestProcessor):
    """
    Request processor that augments a the last message of a chat completion request with
    hallucinations info from the hallucinations intrinsic.
    """

    def __init__(self, io_proc: HallucinationsIOProcessor):
        """
        :param io_proc: IO processor for generating hallucinations.
        """
        self._io_proc = io_proc

    async def aprocess(
        self, inputs: ChatCompletionInputs
    ) -> list[ChatCompletionInputs]:
        # Generate one or more versions of the last turn.
        new_last_turns = await self._io_proc.acreate_chat_completion(inputs)

        final_results = []
        for result in new_last_turns.results:
            new_messages = inputs.messages.copy()
            new_messages[-1] = result.next_message
            final_results.append(inputs.model_copy(update={"messages": new_messages}))
        return final_results
