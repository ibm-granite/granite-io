# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite process reward model intrinsic.

See model card at https://huggingface.co/ibm-granite/granite-3.3-8b-lora-math-prm
"""

# Standard
import math

# Local
from granite_io.backend.base import Backend
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
    UserMessage,
)


class ProcessRewardModelIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the PRM intrinsic, AKA the Granite 3.3 8B Instruct Math PRM LoRA.
    See model card [here](https://huggingface.co/ibm-granite/granite-3.3-8b-lora-math-prm).

    The model must be prompted with a variant of the Granite 3.3 prompt. The model
    needs the logprobs of the input (i.e. user question and assistant response) to
    calculate the Reward score.

    The processor accepts a chat completion request and returns a completion with a
    0.0 to 1.0 PRM score as a string in the content field.

    Example input to the IO processor's :func:`acreate_chat_completion()` call:

    {
        "messages": [
            {
                "role": "user",
                "content": "Weng earns $12 an hour for babysitting. Yesterday, \
    she just did 50 minutes of babysitting. How much did she earn?"
            },
            {
                "role": "assistant",
                "content": "Weng earns $12 per hour for babysitting. Since she only \
    worked for 50 minutes yesterday, we need to calculate her earnings based on \
    the fraction of an hour she worked.\n\nFirst, convert 50 minutes into hours:\n\

    [ 50 \\text{ minutes} = \
    \\frac{50}{60} \\text{ hours} = \\frac{5}{6} \\text{ hours} \\]

    \n\nNext, calculate her \
    earnings by multiplying the fraction of an hour by her hourly rate:\n\

    [ \\text{Earnings} \
    = 12 \\times \\frac{5}{6} \\]

    \n\nPerform the multiplication:\n\

    [ \\text{Earnings} = 12 \
    \\times \\frac{5}{6} = 12 \\times 0.8333\\ldots = 10 \\]

    \n\nTherefore, Weng earned $10 \
    for 50 minutes of babysitting."
            }
        ],
        "generate_inputs": {
            "temperature": 0.0,
            "max_tokens": 4096
        }
    }

    Example prompt that the IO processor would send to the model after adding artificial
    PRM turns:

    "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\n \
    Today's Date: June 23, 2025.\nYou are Granite, developed by IBM. \
    You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user\
    <|end_of_role|>Weng earns $12 an hour for babysitting. \
    Yesterday, she just did 50 minutes of babysitting. \
    How much did she earn? Weng earns \
    $12 per hour for babysitting. Since she only worked for 50 minutes yesterday, \
    we need to calculate her earnings based on the fraction of an hour she worked. 
    Is this response correct so far (Y/N)?<|end_of_text|>\n<|start_of_role|>assistant\
    <|end_of_role|>Y<|end_of_text|>..."

    With continued turns of steps and "Is this response correct so far (Y/N)?\
    <|end_of_text|>\n<|start_of_role|>assistant\
    <|end_of_role|>Y<|end_of_text|>"
    

    Example of processed output from this IO processor for the above raw model output:

    {
        "results": [
            {
                "next_message": {
                    "role": "assistant",
                    "content": "0.989"
                }
            }
        ]
    }
    """

    def __init__(self, backend):
        super().__init__(backend=backend)

        # I/O processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point3InputProcessor()

        # PRM specific tokens of interest
        self.generation_prompt = "Is this response correct so far (Y/N)?"
        self.correct_token = "Y"

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # formats inputs to what the PRM requires:
        # turns of response + generation_prompt + assistant turn

        # first, take the incoming inputs and break them into
        # user question and assistant response
        # create copy of inputs so that we dont modify in place
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())
        messages = inputs.messages

        assert len(messages) <= 3, (
            "PRM BoN only implemented for single turn at the moment"
        )

        user_message = None
        assistant_response = None
        system_message = None

        for message in messages:
            if message.role == "user":
                assert user_message is None, (
                    "multiple user messages: PRM BoN only implemented for single turn"
                )
                user_message = message.content
            elif message.role == "assistant":
                assert assistant_response is None, (
                    "multiple user messages: PRM BoN only implemented for single turn"
                )
                assistant_response = message.content
            elif message.role == "system":
                system_message = message

        assert user_message is not None, "No user input/question found"
        assert assistant_response is not None, "No assistant response found"

        # break assistant response into steps,
        # format with generation prompt to send to backend
        response_steps = assistant_response.split("\n\n")
        if len(response_steps) == 1:
            # no "\n\n" in generation, split on single newline
            response_steps = assistant_response.split("\n")

        assert len(response_steps) > 0, "No steps found for scoring"

        # create a ChatCompletionInputs object with
        # the correct formatted messages
        formatted_messages = [system_message] if system_message is not None else []

        for s_idx, step in enumerate(response_steps):
            if s_idx == 0:
                formatted_messages.append(
                    UserMessage(
                        content=user_message + " " + step + " " + self.generation_prompt
                    )
                )
            else:
                formatted_messages.append(
                    UserMessage(content=step + " " + self.generation_prompt)
                )

            # append the last asst turn
            formatted_messages.append(AssistantMessage(content=self.correct_token))

        # update the inputs to contain the list of
        # newly formatted messages instead of the
        # original user-response pair
        inputs = inputs.with_messages(formatted_messages)
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        # update the prompt with the echo and logprobs
        # (reference: https://github.com/vllm-project/vllm/issues/6508)
        prompt = self.base_input_processor.transform(inputs, False)
        if inputs.generate_inputs is not None:
            result = inputs.generate_inputs.model_copy(
                update={
                    "prompt": prompt,
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 1,
                }
            )
        else:
            inputs.generate_inputs = GenerateInputs()
            result = inputs.generate_inputs.model_copy(
                update={
                    "prompt": prompt,
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 1,
                }
            )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        # Modified OpenAI backend `process_output` to include tokens and logprobs

        # get the scores from the PRM logprobs
        results = []

        assistant_turn_string = "<|start_of_role|>assistant<|end_of_role|>"

        for raw_result in output.results:
            correct_token_probs = []
            print(raw_result.token_logprobs)
            for i, (token, logprob) in enumerate(
                zip(raw_result.tokens, raw_result.token_logprobs, strict=True)
            ):
                if token == self.correct_token:
                    try:
                        if (
                            raw_result.tokens[i - 3]
                            + raw_result.tokens[i - 2]
                            + raw_result.tokens[i - 1]
                            == assistant_turn_string
                        ):
                            # get probabilites by taking the exponent of logprobs
                            correct_token_probs.append(math.exp(logprob))
                    except IndexError:
                        continue
            assert len(correct_token_probs) > 0, (
                "No assistant turns with correct token found"
            )

            prm_score = sum(correct_token_probs) / len(correct_token_probs)

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        content=f"{prm_score:0.3f}",
                        raw=raw_result.completion_string,
                    )
                )
            )

        return ChatCompletionResults(results=results)


class AssistantMessageWithScore(AssistantMessage):
    """Extended output format for the :class:`PRMBestOfNCompositeIOProcessor` with
    an extra field for passing through PRM score."""

    prm_score: float | None = None
    """Output of checking this message with the PRM intrinsic."""


class PRMBestOfNCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that generates multiple responses,
    calculates the PRM score for each reponse,
    and returns the response with the highest PRM score
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        lora_backend: Backend,
        include_score: bool = False,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param lora_backend: Backend for running the PRM intrinsic.
        :param include_score: Whether to include PRM score in assistant response
        """
        self._generator = generator
        self._prm = ProcessRewardModelIOProcessor(lora_backend)
        self._include_score = include_score

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Generate one or more completions
        generator_output = await self._generator.acreate_chat_completion(inputs)

        # Run PRM scoring on all completions in parallel
        futures = []
        for result in generator_output.results:
            futures.append(
                self._prm.acreate_chat_completion(
                    inputs.with_next_message(
                        result.next_message
                    ).with_addl_generate_params({"n": 1, "temperature": 0.0})
                )
            )

        # Process results as they come back.
        all_results = []
        for result, future in zip(generator_output.results, futures, strict=True):
            prm_output = await future
            prm_score = float(prm_output.results[0].next_message.content)

            all_results.append({"result": result, "prm_score": prm_score})

        assert len(all_results) == len(generator_output.results)

        # select response with maximum PRM score.
        selected_response = max(all_results, key=lambda d: d["prm_score"])

        if self._include_score:
            # Tack a PRM score onto the assistant message.
            message_with_score = AssistantMessageWithScore.model_validate(
                selected_response["result"].next_message.model_dump()
                | {"prm_score": selected_response["prm_score"]}
            )
            processed_results = [
                selected_response["result"].model_copy(
                    update={"next_message": message_with_score}
                )
            ]
        else:
            processed_results = [selected_response["result"]]

        return ChatCompletionResults(results=processed_results)
