# SPDX-License-Identifier: Apache-2.0


"""
I/O processors for the Granite answer relevance classifier intrinsic
and the Granite answer relevance rewriter intrinsic.
"""

# Standard
import json
import logging

# Third Party
import pydantic

# Local
from granite_io.io.base import (
    Backend,
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

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

CLASSIFIER_INSTRUCTION_TEXT = """answer_relevance"""


REWRITER_INSTRUCTION_TEXT = (
    "Rewrite the response for relevance.\n"
    "The last assistant response is considered not fully relevant to the last "
    "user inquiry due to {answer_relevance_category}: {answer_relevance_analysis}\n"
    "Decide if you agree with this assessment, then act according to the "
    "following instructions: \n"
    "If you disagree with the assessment, provide a verbatim copy of the original "
    "response.  "
    "DO NOT attempt to correct any other perceived defects in the response.\n"
    "If you agree with the assessment, provide an updated response that no longer "
    "fit the label {answer_relevance_category}, by {correction_method}.  "
    "Your response should be entirely based on the provided documents and "
    "should not rely on other prior knowledge. "
    "Your response should be suitable to be directly provided to the user.  "
    "It should NOT contain meta information regarding the original response, "
    "its assessment, or this instruction.  The user does not see any of these.  "
    "Your response is the only response they will see to their inquiry, "
    "in place of the original response."
)


relevant_categories = [
    "Pertinent",
    "Pertinent with relevant extra",
    "Relevant but not complete",
]

correction_methods = {
    "Excessive unnecessary information": (
        "removing the excessive information from the draft response"
    ),
    "Unduly restrictive": (
        "providing answer without the unwarranted restriction, "
        "or indicating that the desired answer is not available"
    ),
    "Too vague or generic": (
        "providing more crisp and to-the-point answer, "
        "or indicating that the desired answer is not available"
    ),
    "Contextual misalignment": (
        "providing a response that answers the last user "
        "inquiry, taking into account the context of the conversation"
    ),
    "Misinterpreted inquiry": (
        "providing answer only to the correct interpretation of "
        "the inquiry, or attempting clarification if the inquiry is ambiguous "
        "or otherwise confusing, or indicating that the desired answer "
        "is not available"
    ),
    "No attempt": (
        "providing a relevant response if an inquiry should be answered, "
        "or providing a short response if the last user utterance "
        "contains no inquiry"
    ),
}

equivalents = {"Misinterpreted inquiry": ["Misinterpreted question"]}

for k, vs in equivalents.items():
    for v in vs:
        correction_methods[v] = correction_methods[k]

irrelevant_categories = list(correction_methods.keys())


class AnswerRelevanceRawOutput(pydantic.BaseModel):
    answer_relevance_analysis: str
    answer_relevance_category: str
    answer_relevance_judgment: bool


class AnswerRelevanceRewriteRawOutput(pydantic.BaseModel):
    answer_relevance_rewrite: str


CLASSIFIER_RAW_OUTPUT_JSON_SCHEMA = AnswerRelevanceRawOutput.model_json_schema()
REWRITER_RAW_OUTPUT_JSON_SCHEMA = AnswerRelevanceRewriteRawOutput.model_json_schema()


class AnswerRelevanceIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the answer relevance intrinsic, also known as the LoRA Adapter for
    Answer Relevance Classification
    Takes as input a chat completion and returns a completion with an additional user
    message containing json object with fields
        {answer_relevance_judgment, answer_relevance_category,
        answer_relevance_analysis}.
    The input can optionally have documents,
    which are omitted from the the prompt to the model.

    The prompt created for the model is of the following form:
    ```
    <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>....<|end_of_text|>
    ...
    <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>answer_relevance<|end_of_text|>
    ```

    Raw output from the model is in the following form:
    ```
    {
        answer_relevance_analysis: str
        answer_relevance_category: str
        answer_relevance_judgment: bool
    }
    ```

    """

    def __init__(self, backend: Backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point3InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        logger.debug(
            "Classifier input messages\n%s",
            _format_messages_for_display(inputs.messages),
        )

        # Check for the invariants that the model expects its input to satisfy

        if not inputs.messages[-1].role == "assistant":
            raise ValueError("Last message is not an assistant message")
        if not inputs.messages[-2].role == "user":
            raise ValueError("Next to last message is not a user message")

        updated_inputs = inputs.model_copy()

        ## Remove documents if any
        if updated_inputs.documents:
            updated_inputs.documents = []

        # The beginning of the prompt doesn't change relative to base Granite 3.3
        prompt = self.base_input_processor.transform(updated_inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            prompt = (
                prompt + "<|start_of_role|>user<|end_of_role|>"
                f"{CLASSIFIER_INSTRUCTION_TEXT}<|end_of_text|>"
            )
        logger.debug(f"Classifier prompt\n{prompt}")

        generate_inputs_before = (
            inputs.generate_inputs if inputs.generate_inputs else GenerateInputs()
        )
        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Ensure enough tokens for analysis
                "max_tokens": 1024,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_json": CLASSIFIER_RAW_OUTPUT_JSON_SCHEMA},
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        ### To do:
        results = []
        for raw_result in output.results:
            raw_str = raw_result.completion_string
            try:
                result_json = json.loads(raw_str)
                # result_json = {key: result_json[val] for key,val in field_map.items()}
                ## Reset judgment according to category
                category = result_json["answer_relevance_category"]
                if category in relevant_categories:
                    result_json["answer_relevance_judgment"] = True
                elif category in irrelevant_categories:
                    result_json["answer_relevance_judgment"] = False
            except ValueError:
                logger.warning(f"Failed to parse as json string: {raw_str}")
                result_json = {}

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(content=json.dumps(result_json))
                )
            )

        return ChatCompletionResults(results=results)


# def format_rewriter_instruction(assessment: dict):
#     assert isinstance(assessment, dict)
#     for key in ["answer_relevance_analysis", "answer_relevance_category"]:
#         value = assessment.get(key)
#         if not value:
#             logger.warning(f"Incomplete assessment.  Empty key {key} in {assessment}")
#             return None
#     category = assessment["answer_relevance_category"]
#     correction_method = correction_methods.get(category)
#     if correction_method is None:
#         logger.warning(f"Unknown category: {category}")
#         # correction_method = "fixing the error"
#         return None

#     assessment["correction_method"] = correction_method
#     logger.info(
#         f"format_rewriter_instruction\n{json.dumps(assessment, indent=2)}"
#     )
#     rewriter_instruction = REWRITER_INSTRUCTION_TEXT.format(**assessment)
#     return rewriter_instruction


def format_rewriter_instruction(assessment: dict):
    assert isinstance(assessment, dict)
    category = assessment["answer_relevance_category"]
    correction_method = correction_methods.get(category, "fixing the defect")
    assessment["correction_method"] = correction_method
    logger.info(f"format_rewriter_instruction\n{json.dumps(assessment, indent=2)}")
    rewriter_instruction = REWRITER_INSTRUCTION_TEXT.format(**assessment)
    return rewriter_instruction


class AnswerRelevanceRewriteIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the answer relevance rewrite intrinsic, also known as the LoRA
    Adapter for Answer Relevance Rewrite
    Takes as input a chat completion where the second to last message is a candidate
    assistant response, and the last message is an assistant message containing
    assessment of relevance of the candidate response, in the form of  json object
    with fields
    {
        answer_relevance_judgment,
        answer_relevance_category,
        answer_relevance_analysis
    }
    Returns a chat completion where these last messages are replaced by a
    rewritten assistant message.

    Example raw input:
    ```
    <|start_of_role|>documents<|end_of_role|> ... <|end_of_text|>
    <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>....<|end_of_text|>
    ...
    <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>{REWRITER_INSTRUCTION_TEXT}<|end_of_text|>
    ```

    Raw output is the rewritten message.

    """

    def __init__(self, backend: Backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point3InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())
        logger.debug(
            f"Rewriter input messages\n{_format_messages_for_display(inputs.messages)}"
        )

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.documents:
            raise ValueError("Input does not contain documents")
        if not inputs.messages[-1].role == "assistant":
            raise ValueError("Last message is not an assistant message")
        if not inputs.messages[-2].role == "user":
            raise ValueError("Next to last message is not a user message")
        try:
            assessment = json.loads(inputs.messages[-1].content)
        except ValueError as e:
            raise ValueError("Last message cannot be parsed as a json object") from e

        # The beginning of the prompt doesn't change relative to base Granite 3.3
        updated_inputs = inputs.copy()
        updated_inputs.messages = updated_inputs.messages[:-2]
        prompt = self.base_input_processor.transform(updated_inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            rewriter_instruction = format_rewriter_instruction(assessment)
            if not rewriter_instruction:
                logger.info(_format_messages_for_display(inputs.messages))
                raise ValueError(
                    "Assessments is not sufficient to format rewriter input"
                )
            prompt = (
                prompt + f"<|start_of_role|>user<|end_of_role|>{rewriter_instruction}"
                "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            )

        generate_inputs_before = (
            inputs.generate_inputs if inputs.generate_inputs else GenerateInputs()
        )
        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Ensure enough tokens for analysis
                "max_tokens": 1024,
                # Enable constrained decoding on vLLM backends
                "extra_body": {"guided_json": REWRITER_RAW_OUTPUT_JSON_SCHEMA},
            }
        )

        logger.debug("Rewriter prompt\n%s", result.prompt)

        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        ### To do:
        results = []
        for raw_result in output.results:
            raw_str = raw_result.completion_string

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(content=str(raw_str))
                )
            )

        return ChatCompletionResults(results=results)


class AnswerRelevanceCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that checks a RAG request for answer relevance with the
    target response, assess it for relevance, and rewrite it if necessary.
    """

    def __init__(
        self,
        classifier_backend: Backend,
        rewriter_backend: Backend,
    ):
        """
        :param classifier: IO processor that classifies response relevance.
            Should return json object with fields
            {   answer_relevance_analysis,
                answer_relevance_category,
                answer_relevance_judgment}
        :param rewriter: I/O processor that rewrites response for relevance.
        """

        assert isinstance(classifier_backend, Backend)
        assert isinstance(rewriter_backend, Backend)

        self._classifier = AnswerRelevanceIOProcessor(backend=classifier_backend)
        self._rewriter = AnswerRelevanceRewriteIOProcessor(backend=rewriter_backend)

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        logger.debug(
            "Composite classifier input message\n%s",
            _format_messages_for_display(inputs.messages),
        )

        # Run clssifier
        classifier_output = await self._classifier.acreate_chat_completion(inputs)

        classifier_output_message_content = classifier_output.results[
            0
        ].next_message.content
        logger.debug(
            "Composite classifier output message content\n%s",
            classifier_output_message_content,
        )

        # Parse classifier output
        try:
            classifier_output_json = json.loads(classifier_output_message_content)
        except:
            logger.info("Failed to parse classifier_output_message_content as json")
            raise

        logger.debug("Composite classifier output json\n%s", classifier_output_json)

        # Original assistant response already relevant.  Return original response
        classifier_judgment = classifier_output_json["answer_relevance_judgment"]
        if classifier_judgment == "yes" or classifier_judgment is True:
            logger.info("Answer is already relevant.  Do not invoke rewriter")
            chat_output = classifier_output.copy()
            logger.debug(
                "Composite classifier output result\n%s", classifier_output.results[0]
            )
            response = inputs.messages[-1].content
            logger.debug("Original response content\n%s", response)
            chat_output.results[0].next_message.content = json.dumps(
                {"answer_relevance_rewrite": response}
            )
            logger.debug(
                "Composite classifier output result\n%s", classifier_output.results[0]
            )
            return chat_output

        # Original assistant response is irrelevant.
        # Invoke rewriter to generate a rewritten response
        logger.debug("Answer is not relevant.  Invoke rewriter")

        classifier_output_json_str = json.dumps(classifier_output_json, indent=2)
        logger.debug(
            "Composite classifier_output_json_str\n%s", classifier_output_json_str
        )

        rewriter_input = Granite3Point3Inputs.model_validate(
            {
                "messages": inputs.messages
                + [
                    {"role": "user", "content": "answer_relevance"},
                    {"role": "assistant", "content": classifier_output_json_str},
                ],
                "documents": inputs.documents,
                "generate_inputs": {
                    "temperature": 0.0  # Ensure consistency across runs
                },
            }
        )

        logger.debug(
            "Composite rewriter input messages\n%s",
            _format_messages_for_display(rewriter_input.messages),
        )

        chat_output = await self._rewriter.acreate_chat_completion(rewriter_input)

        logger.info(f"Composite rewriter output\n{chat_output}")

        return chat_output


def _format_messages_for_display(messages):
    return "\n".join([f"{message.role}: {message.content}" for message in messages])
