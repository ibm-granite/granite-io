# SPDX-License-Identifier: Apache-2.0
"""
I/O processor for the Granite context relevancy intrinsic.
"""

# Standard
from enum import Enum
import json
import re
import warnings

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

# Regex for v3.3 constrained decoding
_JSON_FENCE_REGEX = (
    r"```json\n\{\s*\"context_relevance\"\s*:\s*\"(irrelevant|relevant|"
    r"partially relevant)\"\s*\}\n```"
)

INSTRUCTION_TEXT = (
    "Analyze the provided document in relation to the final user query from the "
    "conversation. Determine if the document contains information that could help "
    "answer the final user query. Output 'relevant' if the document contains "
    "substantial information directly useful for answering the final user query. "
    "Output 'partially relevant' if the document contains some related information "
    "that could partially help answer the query, or if you are uncertain about the "
    "relevance - err on the side of 'partially relevant' when in doubt. "
    "Output 'irrelevant' only if the document clearly contains no information that "
    "could help answer the final user query. When uncertain, choose 'partially "
    "relevant' rather than 'irrelevant'."
)

json_object = {"context_relevance": "YOUR_CONTEXT_RELEVANCE_CLASSIFICATION_HERE"}
json_str = json.dumps(json_object, indent=4)
JSON = (
    "Your output should be a JSON structure with the context relevance "
    "classification:\n" + "```json\n" + json_str + "\n```"
)

FINAL_QUERY_ROLE = (
    "<|start_of_role|>final_user_query<|end_of_role|>{final_user_query}"
    "<|end_of_text|>\n"
)

# Set up invocation prompt for context relevancy LoRA
INVOCATION_PROMPT = (
    "<|start_of_role|>context_relevance: "
    + INSTRUCTION_TEXT
    + " "
    + JSON
    + "<|end_of_role|>"
)


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

    Example raw input:
    ```
    <|start_of_role|>user<|end_of_role|>How is enterprise value calculated?
    <|end_of_text|>
    <|start_of_role|>final_user_query<|end_of_role|>How is enterprise value
    calculated?<|end_of_text|>
    <|start_of_role|>document {"document_id": "1"}<|end_of_role|>

    You wouldn't know it's value (Enterprise Value) without knowing its cash balance.
    The equation:   EV = Market Cap + Minority Interest + Preferred Stock + Debt - Cash
    Enterprise Value is the value of the company to ALL shareholders (creditors,
    preferred stock holders, common stock holders). So, taking on debt could either
    increase or decrease the EV depending on the cash balance of the company.
    This will have no effect, directly, on the market cap.
    It will, however effect the present value of its future cash flows as the WACC will
    increase due to the new cost of debt (interest payments, higher risk of bankruptcy,
    less flexibility by management).<|end_of_text|>
    <|start_of_role|>context_relevance: Analyze the provided document in relation to
    the final user query from the conversation.
    Determine if the document contains information that could help answer the final
    user query.
    Output 'relevant' if the document contains substantial information directly useful
    for answering the final user query.
    Output 'partially relevant' if the document contains some related information that
    could partially help answer the query, or if you are uncertain about the relevance
    - err on the side of 'partially relevant' when in doubt.
    Output 'irrelevant' only if the document clearly contains no information that could
    help answer the final user query.
    When uncertain, choose 'partially relevant' rather than 'irrelevant'.
    Your output should be a JSON structure with the context relevance classification:
    ```json
    {
        "context_relevance": "YOUR_CONTEXT_RELEVANCE_CLASSIFICATION_HERE"
    }
    ```
    <|end_of_role|>
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

        # If the last message is from the assistant, we raise an error.
        # We only want the last turn to be the user message
        if inputs.messages[-1].role != "user":
            raise ValueError("Last message is not a user message")

        # Create a copy of inputs without documents for the base processor
        inputs_without_docs = inputs.model_copy(update={"documents": []})

        # The beginning of the prompt doesn't change relative to base Granite 3.3
        # but we don't pass documents to the base processor as we change the
        # document order in the prompt
        prompt_prefix = self.base_input_processor.transform(inputs_without_docs, False)

        # Add the final user message to the prompt to remove any confusion about
        # the final user query to the model
        prompt = prompt_prefix + FINAL_QUERY_ROLE.format(
            final_user_query=inputs.messages[-1].content
        )

        # Add the document after the final user query so that the model can see
        # the final user query and the document together to determine relevance
        document = inputs.documents[0]
        document_role = (
            f'<|start_of_role|>document {{"document_id": "1"}}<|end_of_role|>\n'
            f"{document.text}<|end_of_text|>\n"
        )
        prompt = prompt + document_role

        # We add the generation prompt to the end of the prompt
        if add_generation_prompt:
            prompt = prompt + INVOCATION_PROMPT

        return inputs.generate_inputs.model_copy(
            update={
                "prompt": prompt,
                "max_tokens": 100,
                "extra_body": {"guided_regex": _JSON_FENCE_REGEX},
            }
        )

    def output_to_result(
        self, output: GenerateResults, inputs: ChatCompletionInputs | None = None
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            raw_str = raw_result.completion_string.strip()
            relevance_label = None

            # Try to parse JSON and extract context_relevance
            try:
                # Look for JSON structure in the output
                match = re.search(r"\{.*\}", raw_str, re.DOTALL)
                parsed_output = ContextRelevanceRawOutput.model_validate_json(
                    match.group(0)
                )
                relevance_label = parsed_output.context_relevance
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: check if any of the valid labels are in the raw string
            if not relevance_label:
                if CRLabel.PARTIAL in raw_str:
                    relevance_label = CRLabel.PARTIAL
                elif CRLabel.IRRELEVANT in raw_str:
                    relevance_label = CRLabel.IRRELEVANT
                elif CRLabel.RELEVANT in raw_str:
                    relevance_label = CRLabel.RELEVANT

            # If we found a valid label, use it; otherwise use raw string
            content = relevance_label if relevance_label else "ERROR"

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(content=content, raw=raw_str)
                )
            )
        return ChatCompletionResults(results=results)


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
    ):
        """
        :param generator: I/O processor that generates the results that this I/O
         processor should validate.
        :param context_relevancy: IO processor for the context relevancy model.
         Should return a string indicating the document's relevance
        """
        self._generator = generator
        self._context_relevancy = context_relevancy

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
                        {"temperature": 0.0, "n": 1}
                    )
                )
            )

        # Process results as they come back. Check each document's relevancy.
        relevant_documents = []
        for document, future in zip(inputs.documents, futures, strict=True):
            relevancy_output_obj = await future
            context_relevancy_output = relevancy_output_obj.results[0]
            if context_relevancy_output.next_message.content in (
                CRLabel.RELEVANT,
                CRLabel.PARTIAL,
            ):
                # Document is relevant to the last user question; keep it
                relevant_documents.append(document)

        # Warn if all documents were filtered out
        if not relevant_documents and inputs.documents:
            warnings.warn(
                f"All {len(inputs.documents)} documents were classified as irrelevant "
                "and removed by the context relevancy filter. The generator will "
                "receive no context documents.",
                UserWarning,
                stacklevel=2,
            )

        # Update the inputs with the relevant documents (can be empty)
        inputs_with_updated_docs = inputs.model_copy(
            update={"documents": relevant_documents}
        )

        return await self._generator.acreate_chat_completion(inputs_with_updated_docs)
