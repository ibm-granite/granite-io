# SPDX-License-Identifier: Apache-2.0


"""
I/O processor for the Granite answerability intrinsic.
"""

# Local
from granite_io.io.base import ModelDirectInputOutputProcessor
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2InputProcessor,
    Granite3Point2Inputs,
)
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateResults,
)


class AnswerabilityIOProcessor(ModelDirectInputOutputProcessor):
    """
    I/O processor for the answerability intrinsic, also known as the LoRA Adapter for
    Answerability Classification
    Takes as input a chat completion and returns a completion with a 1/0 answerability
    flag as a string in the content field.

    Example raw input:
    ```
    <|start_of_role|>documents<|end_of_role|> ... <|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Can I go tubing along the Russian River in
    Sonoma County?<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>Yes, tubing
    along the Russian River in Sonoma County is a popular activity. The recommended
    route for tubing is the 2-mile stretch from Steelhead Beach to Sunset Beach in
    Forestville.<|end_of_text|> <|start_of_role|>user<|end_of_role|>How can I arrange
    transportation between Steelhead Beach and Sunset Beach?<|end_of_text|>
    <|start_of_role|>answerability<|end_of_role|>
    ```

    Raw output should be either `answerable` or `unanswerable`.
    """

    def __init__(self, backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point2InputProcessor()

    def inputs_to_string(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        # Validate the input and convert to Granite input
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.documents:
            raise ValueError("Input does not contain documents")
        if not inputs.messages[-1].role == "user":
            raise ValueError("Last message is not a user message")

        # The beginning of the prompt doesn't change relative to base Granite 3.2
        prompt_prefix = self.base_input_processor.transform(inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            return prompt_prefix + "<|start_of_role|>answerability<|end_of_role|>"
        return prompt_prefix

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            if raw_result.completion_string in ("answerable", "unanswerable"):
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=raw_result.completion_string
                        )
                    )
                )
            else:
                # Improper output. Pass through as an error message for now instead of
                # raising an exception from the I/O processor.
                results.append(
                    ChatCompletionResult(next_message=AssistantMessage(content="ERROR"))
                )
        return ChatCompletionResults(results=results)
