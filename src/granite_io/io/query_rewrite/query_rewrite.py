# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite query rewrite intrinsic.
"""

# Standard
import json

# Local
from granite_io.io.base import ModelDirectInputOutputProcessorWithGenerate
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
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
)

INSTRUCTION_TEXT = (
    "Reword the final utterance from the USER into a single utterance that doesn't "
    "need the prior conversation history to understand the user's intent. If the final "
    "utterance is a clear and standalone question, please DO NOT attempt to rewrite "
    "it, rather output the last user utterance as is. "
)
JSON = 'Your output format should be in JSON: { "rewritten_question": <REWRITE> }'
REWRITE_PROMPT = (
    "<|start_of_role|>rewrite: " + INSTRUCTION_TEXT + JSON + "<|end_of_role|>"
)


class QueryRewriteIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the query rewrite intrinsic, also known as the [LoRA Adapter for 
    Query Rewrite](https://huggingface.co/ibm-granite/
    granite-3.2-8b-lora-rag-query-rewrite)

    Takes as input a chat completion and returns a completion with a rewrite of the
    most recent user turn (last in the conversation).

    Example raw input:
    ```
    <|start_of_role|>user<|end_of_role|>Tim Cook is the CEO of Apple Inc.<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>Yes, Tim Cook is the Chief Executive \
Officer of Apple Inc.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>and for Microsoft?<|end_of_text|>
    ```
    
    Example of corresponding raw output:
    ```
    { "rewritten_question": "Who is the CEO of Microsoft" }
    ```

    Output string with the rewrite of the last user turn:
        'Who is the CEO of Microsoft'
    """

    def __init__(self, backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point2InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.messages[-1].role == "user":
            raise ValueError("Last message is not a user message")

        # The beginning of the prompt doesn't change relative to base Granite 3.2
        prompt = self.base_input_processor.transform(inputs, False)

        # To invoke the model, we add the rewrite prompt to the prompt prefix:
        if add_generation_prompt:
            prompt = prompt + REWRITE_PROMPT
        result = inputs.generate_inputs.model_copy(
            update={
                "prompt": prompt,
                "max_tokens": 80,
                # TODO Enable constrained decoding on vLLM backends
                # "extra_body": {
                #     "guided_regex": r"\d{1,2}%",
                # },
            }
        )
        return result

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,  # pylint: disable=unused-argument
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            json_result = raw_result.completion_string
            try:
                rewrite = json.loads(json_result)["rewritten_question"]
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Cannot parse the JSON. Pass through the unparsed raw output for now
                # instead of raising an exception from the I/O processor.
                print(f"\nException: {e}\n")
                rewrite = json_result

            results.append(
                ChatCompletionResult(next_message=AssistantMessage(content=rewrite))
            )

        return ChatCompletionResults(results=results)
