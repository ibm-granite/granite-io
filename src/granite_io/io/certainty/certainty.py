# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite certainty intrinsic.

See model card at https://huggingface.co/ibm-granite/granite-uncertainty-3.2-8b-lora
"""

# Local
from granite_io.backend.base import Backend
from granite_io.io.base import (
    InputOutputProcessor,
    ModelDirectInputOutputProcessorWithGenerate,
)
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


class CertaintyIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the certainty intrinsic, AKA the Granite 3.2 8B Instruct
    Uncertainty LoRA. See model card [here](
        https://huggingface.co/ibm-granite/granite-uncertainty-3.2-8b-lora).
        
    The model must be prompted with a variant of the Granite 3.2 prompt. The model needs
    constrained decoding to produce a reliable result. The first token of the model
    output will be a certainty score encoded as a number from 0 to 9, inclusive.
    
    The processor as input a chat completion request and returns a completion with a 0.0
    to 0.9 certainty amount as a string in the content field.
    
    Example input to the IO processor's :func`acreate_chat_completion()` call: 
    
    ```
    {
        "messages": [
            {
                "role": "user", "content": "Can I go tubing along the Russian River in
                Sonoma County?"
            }
            {
                "role": "assistant", "content": "Yes, tubing along the Russian River in
                Sonoma County is a "
                           "popular activity. The recommended route for tubing is the "
                           "2-mile stretch from Steelhead Beach to Sunset Beach in "
                           "Forestville."
            },
        ], "documents": [
            {
                "text": "Sonoma County has many vacation destinations..."
            }, {
                "text": "Marin County also has vacation destinations..."
            }
        ], "generate_inputs": {
            "temperature": 0.0, "max_tokens": 4096,
        }
    }
    ```
    
    Example prompt that the IO processor would send to the model if it received the
    above input: ``` <|start_of_role|>documents<|end_of_role|> ... <|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Can I go tubing along the Russian River in \
    Sonoma County?<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>Yes, tubing
    along the Russian River in Sonoma County is a popular activity. The recommended
    route for tubing is the 2-mile stretch from Steelhead Beach to Sunset Beach in
    Forestville.<|end_of_text|> <|start_of_role|>certainty<|end_of_role|>```
    
    Example of raw output of the model for the above request: 
    ```
    7<|end_of_text|>
    ```
    (note that constrained decoding is required to produce the `<|end_of_text|>` token 
    reliably)
    
    Example of processed output from this IO processor for the above raw model output:
    ```
    {
        "results": [
            {
                "next_message": {
                    "role": "assistant",
                    "content": "0.7"
                }
            }
        ]
    }
    ```
    """

    def __init__(self, backend):
        super().__init__(backend=backend)

        # I/O processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point2InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if inputs.messages[-1].role not in ("user", "assistant"):
            raise ValueError("Last message is not a user or assistant message")

        # The beginning of the prompt doesn't change relative to base Granite 3.2
        prompt = self.base_input_processor.transform(inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            prompt = prompt + "<|start_of_role|>certainty<|end_of_role|>"
        result = inputs.generate_inputs.model_copy(
            update={
                "prompt": prompt,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_choice": [str(i) for i in range(10)]},
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            # Raw output should be a number from 0-9
            raw_str = raw_result.completion_string

            # Ignore everything after first character
            raw_number = raw_str[0]
            if raw_number.isdigit():
                parsed_prob = 0.1 * int(raw_number)
            else:
                parsed_prob = float("nan")  # Return nan on parse failure
            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        # Round to 1 digit so that we don't get 0.7 => 0.70000000001
                        content=f"{parsed_prob:0.1f}",
                        _raw=raw_str,
                    )
                )
            )

        return ChatCompletionResults(results=results)


DEFAULT_CANNED_RESPONSE = (
    "Sorry, but I am unable to generate a response with high certainty."
)


class AssistantMessageWithScore(AssistantMessage):
    """Extended output format for the :class:`CertaintyCompositeIOProcessor` with
    an extra field for passing through certainty score."""

    certainty_score: float | None = None
    """Output of checking this message with the certainty intrinsic."""


class CertaintyCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that generates a response, checks the response for
    certainty, and falls back on a canned response if the check falls below a threshold.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        lora_backend: Backend,
        threshold: float = 0.75,
        canned_response: str = DEFAULT_CANNED_RESPONSE,
        include_score: bool = False,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param lora_backend: Backend for running the certainty intrinsic.
        :param threshold: If the certainty score of every completion is below this
         value, a canned response will be given.
        :param canned_response: Fallback response if none of the responses from the
         generator pass the threshold.
        """
        self._generator = generator
        self._certainty = CertaintyIOProcessor(lora_backend)
        self._threshold = threshold
        self._canned_response = canned_response
        self._include_score = include_score

    def update_threshold(self, threshold: float):
        """Convenience method to update the filtering threshold.

        :param threshold: New value to be applied to subsequent calls to the certainty
         I/O processor."""
        self._threshold = threshold

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Generate one or more completions
        generator_output = await self._generator.acreate_chat_completion(inputs)

        # Run certainty checks on all completions in parallel
        futures = []
        for result in generator_output.results:
            futures.append(
                self._certainty.acreate_chat_completion(
                    inputs.with_next_message(
                        result.next_message
                    ).with_addl_generate_params({"n": 1, "temperature": 0.0})
                )
            )

        # Process results as they come back. Check each certainty score against the
        # threshold.
        processed_results = []
        max_certainty = 0.0
        for result, future in zip(generator_output.results, futures, strict=True):
            certainty_output = await future
            certainty_score = float(certainty_output.results[0].next_message.content)
            max_certainty = max(certainty_score, max_certainty)
            if certainty_score >= self._threshold:
                # Tack a certainty score onto the assistant message.
                message_with_score = AssistantMessageWithScore.model_validate(
                    result.next_message.model_dump()
                    | {"certainty_score": certainty_score}
                )
                processed_results.append(
                    result.model_copy(update={"next_message": message_with_score})
                )

        if len(processed_results) == 0:
            # No completions passed the certainty threshold. Use canned response.
            processed_results.append(
                ChatCompletionResult(
                    next_message=AssistantMessageWithScore(
                        content=self._canned_response, certainty_score=max_certainty
                    )
                )
            )

        return ChatCompletionResults(results=processed_results)
