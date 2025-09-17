# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import re

# Third Party
import pydantic

# Local
from granite_io.io.base import ModelDirectInputOutputProcessorWithGenerate
from granite_io.io.granite_3_3.input_processors import (
    granite_3_3_input_processor as g33_input_processor,
)
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)

# Granite 3.3 specific configurations
# Regex pattern for constrained decoding with JSON fence format
_JSON_FENCE_REGEX = r"```json\n\{\s*\"rewritten_question\"\s*:\s*\"[^\"]*\"\s*\}\n```"

# JSON object template for Granite 3.3 output format
_JSON_OBJECT_3_3 = {"rewritten_question": "YOUR_REWRITTEN_QUESTION_HERE"}

JSON_OBJECT_3_3_STR = json.dumps(_JSON_OBJECT_3_3, indent=4)

# Template for instructing the model on expected JSON output format
JSON_TEMPLATE_3_3 = (
    "Your output should be a JSON structure with the rewritten question:\n"
    "```json\n"
    f"{JSON_OBJECT_3_3_STR}\n"
    "```"
)

# Query to rewrite prompt template for query rewrite LoRA
QUERY_TO_REWRITE_TEMPLATE = (
    "<|start_of_role|>query_to_rewrite<|end_of_role|>{msg}<|end_of_text|>\n"
)

INSTRUCTION_TEXT_3_3 = (
    "Given the conversation history above and the specific query provided in the "
    "'query_to_rewrite' role, rewrite that query into a standalone question that "
    "captures the user's intent without requiring the conversation context. "
    "If the query is already clear and standalone, output it as is."
)

REWRITE_PROMPT_3_3 = (
    "<|start_of_role|>rewrite: "
    + INSTRUCTION_TEXT_3_3
    + "\n"
    + JSON_TEMPLATE_3_3
    + "<|end_of_role|>"
)


class QueryRewriteRawOutput(pydantic.BaseModel):
    rewritten_question: str


class QueryRewriteIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for Granite 3.3 query-rewrite LoRAs.

    This processor takes a conversation history and rewrites the final user query
    into a standalone question that doesn't require context from the conversation
    history to understand the user's intent.

    Note: Previously supported Granite 3.2, but support has been removed to
    simplify the codebase and focus on the newer 3.3 architecture.
    """

    def __init__(self, backend):
        """
        Initialize the QueryRewriteIOProcessor for Granite 3.3.

        Args:
            backend: The model backend to use for generation
        """
        super().__init__(backend=backend)

        # Import and configure Granite 3.3 specific components
        Granite3Point3InputProcessor = g33_input_processor.Granite3Point3InputProcessor
        Granite3Point3Inputs = g33_input_processor.Granite3Point3Inputs

        self.InputsModel = Granite3Point3Inputs
        self.base_input_processor = Granite3Point3InputProcessor()
        # Use guided regex for constrained JSON output
        self._extra_body = {"guided_regex": _JSON_FENCE_REGEX}

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        """
        Convert chat completion inputs to generation inputs for query rewriting.

        Args:
            inputs: Chat completion inputs containing conversation history
            add_generation_prompt: Whether to add the generation prompt

        Returns:
            GenerateInputs configured for query rewriting task
        """
        # Validate and normalize inputs for Granite 3.3
        inputs = self.InputsModel.model_validate(inputs.model_dump())

        if inputs.messages[-1].role != "user":
            raise ValueError("Last message is not a user message")

        # Build the prompt using conversation history and the query to rewrite
        prompt_prefix = self.base_input_processor.transform(inputs, False)
        prompt = self._make_prompt_3_3(
            prompt_prefix, inputs.messages[-1].content, add_generation_prompt
        )

        return inputs.generate_inputs.model_copy(
            update={
                "prompt": prompt,
                "max_tokens": 256,  # Sufficient for query rewriting tasks
                "extra_body": self._extra_body,
            }
        )

    def _make_prompt_3_3(self, prefix: str, final_msg: str, add_prompt: bool) -> str:
        """
        Build the complete prompt for Granite 3.3 query rewriting.

        Args:
            prefix: Conversation history formatted by input processor
            final_msg: The final user message to be rewritten
            add_prompt: Whether to add the rewrite instruction prompt

        Returns:
            Complete prompt string for the model
        """
        prompt = prefix + QUERY_TO_REWRITE_TEMPLATE.format(msg=final_msg)
        if add_prompt:
            prompt += REWRITE_PROMPT_3_3
        return prompt

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        """
        Process the model output and extract the rewritten query.

        Args:
            output: Raw generation results from the model
            inputs: Original chat completion inputs (for context)

        Returns:
            ChatCompletionResults with rewritten query as the next message
        """
        results = []
        for raw in output.results:
            s = raw.completion_string.strip()
            rewrite = None

            # Attempt to parse JSON output to extract rewritten question
            try:
                m = re.search(r"\{.*\}", s, re.DOTALL)
                parsed_output = QueryRewriteRawOutput.model_validate_json(m.group(0))
                rewrite = parsed_output.rewritten_question
            except (json.JSONDecodeError, AttributeError, pydantic.ValidationError):
                # JSON parsing failed, will fallback to raw string
                pass

            # Fallback to raw string if JSON parsing failed
            if not rewrite:
                rewrite = s

            # Create new message with rewritten content and return it to the caller
            rewritten = inputs.messages[-1].model_copy(update={"content": rewrite})
            results.append(ChatCompletionResult(next_message=rewritten))

        return ChatCompletionResults(results=results)
