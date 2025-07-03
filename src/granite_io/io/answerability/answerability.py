# SPDX-License-Identifier: Apache-2.0


"""
I/O processor for the Granite answerability intrinsic.
"""

# Local
from granite_io.io.base import (
    InputOutputProcessor,
    ModelDirectInputOutputProcessorWithGenerate,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3InputProcessor,
    Granite3Point3Inputs,
)
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)


class AnswerabilityIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
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
        self.base_input_processor = Granite3Point3InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.documents:
            raise ValueError("Input does not contain documents")
        if not inputs.messages[-1].role == "user":
            raise ValueError("Last message is not a user message")

        # The beginning of the prompt doesn't change relative to base Granite 3.3
        prompt = self.base_input_processor.transform(inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            prompt = prompt + "<|start_of_role|>answerability<|end_of_role|>"

        generate_inputs_before = (
            inputs.generate_inputs if inputs.generate_inputs else GenerateInputs()
        )
        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Ensure enough tokens to produce "unanswerable"
                "max_tokens": 16,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_choice": ["answerable", "unanswerable"]},
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            raw_str = raw_result.completion_string
            if raw_str in ("answerable", "unanswerable"):
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=raw_result.completion_string
                        )
                    )
                )
            # Model may forget to generate end of string
            elif raw_str.startswith("answerable"):
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(content="answerable", raw=raw_str)
                    )
                )
            elif raw_str.startswith("unanswerable"):
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content="unanswerable", raw=raw_str
                        )
                    )
                )
            else:
                # Improper output. Pass through as an error message for now instead of
                # raising an exception from the I/O processor.
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(content="ERROR", raw=raw_str),
                    )
                )
        return ChatCompletionResults(results=results)


DEFAULT_CANNED_RESPONSE = (
    "Sorry, but I am unable to answer this question from the documents retrieved."
)


class AnswerabilityCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that checks a RAG request for answerability with the
    target document snippets and triggers an appropriate fallback if the request
    is unanswerable with the target documents.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        answerability: InputOutputProcessor,
        fallback_type: str = "canned_response",
        canned_response: str = DEFAULT_CANNED_RESPONSE,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param answerability_io_proc: IO processor for the answerability model.
         Should return either "answerable" or "unanswerable".
        :param fallback_type: One of "canned_response" or "generate_without_docs"
        :param canned_response: Fallback response to use if ``fallback_type`` is
         "canned_response"
        """
        if fallback_type not in ("canned_response", "generate_without_docs"):
            raise ValueError(
                f"Unknown fallback type '{fallback_type}'. Should be one "
                f"of 'canned_response' or "
                "enerate_without_docs"
            )
        self._generator = generator
        self._answerability = answerability
        self._fallback_type = fallback_type
        self._canned_response = canned_response

    def update_threshold(self, threshold: float):
        """Convenience method to update the filtering threshold.

        :param threshold: New value to be applied to subsequent calls to the certainty
         I/O processor."""
        self._threshold = threshold

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Run a single answerability check
        answerability_output = (
            await self._answerability.acreate_chat_completion(
                inputs.with_addl_generate_params({"temperature": 0.0, "n": 1})
            )
        ).results[0]

        if answerability_output.next_message.content == "answerable":
            # User message is answerable. Return a result.
            return await self._generator.acreate_chat_completion(inputs)
        if self._fallback_type == "generate_without_docs":
            # User message is not answerable; attempt to generate a non-RAG result.
            return await self._generator.acreate_chat_completion(
                inputs.model_copy(update={"documents": None})
            )
        # if unanswerable and self._fallback_type == "canned_response":
        return ChatCompletionResults(
            results=[
                ChatCompletionResult(
                    next_message=AssistantMessage(content=self._canned_response)
                )
            ]
        )
