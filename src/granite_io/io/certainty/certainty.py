# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite certainty intrinsic.

See model card at https://huggingface.co/ibm-granite/granite-uncertainty-3.2-8b-lora
"""

# Local
from granite_io.io.base import ModelDirectInputOutputProcessorWithGenerate
from granite_io.io.granite_3_2.granite_3_2 import Granite3Point2InputOutputProcessor
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    _Granite3Point2Inputs,
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
        self.base_processor = Granite3Point2InputOutputProcessor()

    def inputs_to_generate_inputs(self, inputs: ChatCompletionInputs) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = _Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if inputs.messages[-1].role not in ("user", "assistant"):
            raise ValueError("Last message is not a user or assistant message")

        # The beginning of the prompt doesn't change relative to base Granite 3.2
        prompt_prefix = self.base_processor.inputs_to_string(inputs, False)

        # Only the generation prompt portion changes
        prompt = prompt_prefix + "<|start_of_role|>certainty<|end_of_role|>"
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
