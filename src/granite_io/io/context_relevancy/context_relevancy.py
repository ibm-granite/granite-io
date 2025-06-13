# SPDX-License-Identifier: Apache-2.0


"""
I/O processor for the Granite context relevancy intrinsic.
"""

# Standard
from enum import Enum
import json

# Third Party
import pydantic

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

CONTEXT_RELEVANCE_PROMPT = "<|start_of_role|>context_relevance<|end_of_role|>"


class CRLabel(str, Enum):
    IRRELEVANT = "irrelevant"
    RELEVANT = "relevant"
    PARTIAL = "partially relevant"


class ContextRelevanceRawOutput(pydantic.BaseModel):
    context_relevance: CRLabel


RAW_OUTPUT_JSON_SCHEMA = ContextRelevanceRawOutput.model_json_schema()


class ContextRelevancyIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for the context relevancy intrinsic, also known as the LoRA Adapter
    for Context Relevancy Classification
    Takes as input a chat completion and returns a completion with a context relevancy
    flag as a string in the "" field.

    Example raw input:
    ```
    <|start_of_role|>user<|end_of_role|>How is enterprise value calculated?
    <|end_of_text|>
    <|start_of_role|>documents<|end_of_role|>Document 0

    You wouldn't know it's value (Enterprise Value) without knowing its cash balance.
    The equation:   EV = Market Cap + Minority Interest + Preferred Stock + Debt - Cash
    Enterprise Value is the value of the company to ALL shareholders (creditors,
    preferred stock holders, common stock holders). So, taking on debt could either
    increase or decrease the EV depending on the cash balance of the company.
    This will have no effect, directly, on the market cap.
    It will, however effect the present value of its future cash flows as the WACC will
    increase due to the new cost of debt (interest payments, higher risk of bankruptcy,
    less flexibility by management).<|end_of_text|>
    <|start_of_role|>context_relevance<|end_of_role|>
    ```


    Raw output be in json format:
    ```json
    {
        "context_relevance": "YOUR_CONTEXT_RELEVANCE_CLASSIFICATION_HERE"
    }
    ```
    Where the value in `YOUR_CONTEXT_RELEVANCE_CLASSIFICATION_HERE` can be `irrelevant`,
    `partially relevant`, or `relevant`.

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
            raise ValueError("Input does not contain any documents")
        if len(inputs.documents) > 1:
            raise ValueError("Input contains more than one document")

        updated_inputs = inputs
        # If the last message is from the assistant, remove it;
        # We only want the last turn to be the user message
        if inputs.messages[-1].role != "assistant":
            updated_inputs = inputs.with_messages(inputs.messages[:-1])

        # The beginning of the prompt doesn't change relative to base Granite 3.2
        prompt = self.base_input_processor.transform(updated_inputs, False)

        # Only the generation prompt portion changes
        if add_generation_prompt:
            prompt = prompt + CONTEXT_RELEVANCE_PROMPT

        generate_inputs_before = (
            updated_inputs.generate_inputs
            if updated_inputs.generate_inputs
            else GenerateInputs()
        )
        result = generate_inputs_before.model_copy(
            update={
                "prompt": prompt,
                # Ensure enough tokens to produce the answer
                "max_tokens": 100,
                "extra_body": {
                    "guided_json": RAW_OUTPUT_JSON_SCHEMA,
                },
            }
        )
        return result

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            raw_str = raw_result.completion_string
            json_result = raw_str
            try:
                json_result = json.loads(raw_str)["context_relevance"]
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Cannot parse the JSON. Pass through the unparsed raw output for now
                # instead of raising an exception from the I/O processor.
                print(f"\nException: {e}\n")
                json_result = raw_str
            if json_result in (CRLabel.IRRELEVANT, CRLabel.RELEVANT, CRLabel.PARTIAL):
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=raw_result.completion_string
                        )
                    )
                )
            # Model may forget to generate end of string
            elif CRLabel.PARTIAL in json_result:
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=CRLabel.PARTIAL, raw=raw_str
                        )
                    )
                )
            elif CRLabel.IRRELEVANT in json_result:
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=CRLabel.IRRELEVANT, raw=raw_str
                        )
                    )
                )
            elif CRLabel.RELEVANT in json_result:
                results.append(
                    ChatCompletionResult(
                        next_message=AssistantMessage(
                            content=CRLabel.RELEVANT, raw=raw_str
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
    "Sorry, but I am unable to identify whether the document(s) "
    "is relevant or irrelevant to the last user question."
)


class ContextRelevancyCompositeIOProcessor(InputOutputProcessor):
    """
    Composite I/O processor that only keeps contexts (documents)
    that are relevant or partially relevant to the last user question
    of the RAG request.
    All other documents are filtered out.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        context_relevancy: InputOutputProcessor,
        fallback_type: str = "canned_response",
        canned_response: str = DEFAULT_CANNED_RESPONSE,
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor shoid validate.
        :param context relevancy_io_proc: IO processor for the context relevancy model.
         Should return a string indicating the document's relevance
        :param fallback_type: "canned_response"
        :param canned_response: Fallback response to use if ``fallback_type`` is
         "canned_response"
        """
        if fallback_type not in ("canned_response"):
            raise ValueError(
                f"Unknown fallback type '{fallback_type}'. Should be one "
                f"of 'canned_response'"
            )
        self._generator = generator
        self._context_relevancy = context_relevancy
        self._fallback_type = fallback_type
        self._canned_response = canned_response

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Run context relevancy checks on all documents in parallel
        futures = []
        for document in inputs.documents:
            single_document_input = inputs.model_copy(update={"documents": [document]})
            futures.append(
                self._context_relevancy.acreate_chat_completion(
                    single_document_input.with_addl_generate_params(
                        {"temperature": 0.0}
                    )
                )
            )

        # Process results as they come back. Check each document's relevancy.
        relevant_documents = []
        for document, future in zip(inputs.documents, futures, strict=True):
            relevancy_output_obj = await future
            context_relevancy_output = relevancy_output_obj.results[0]
            if (
                context_relevancy_output.next_message.content == CRLabel.RELEVANT
                or context_relevancy_output.next_message.content == CRLabel.PARTIAL
            ):
                # Document is relevant to the last user question; keep it
                relevant_documents.append(document)

        inputs_with_updated_docs = inputs.model_copy(
            update={"documents": relevant_documents}
        )
        return await self._generator.acreate_chat_completion(inputs_with_updated_docs)
