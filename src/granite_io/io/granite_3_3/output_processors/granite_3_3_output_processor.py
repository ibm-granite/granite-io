# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import uuid

# Third Party
import pydantic

# Local
from granite_io.io.base import OutputProcessor
from granite_io.io.consts import (
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_2B_OLLAMA,
    _GRANITE_3_3_8B_HF,
    _GRANITE_3_3_8B_OLLAMA,
    _GRANITE_3_3_COT_END,
    _GRANITE_3_3_COT_START,
    _GRANITE_3_3_MODEL_NAME,
    _GRANITE_3_3_RESP_END,
    _GRANITE_3_3_RESP_START,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.granite_3_3.output_processors.granite_3_3_output_parser import (
    parse_model_output,
)
from granite_io.io.registry import output_processor
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    FunctionCall,
    GenerateResults,
)

# Some versions of the model are known to shorten "Here is" to "Here's", so we
# provide alternate forms of these strings for those versions.
_COT_START_ALTERNATIVES = [
    _GRANITE_3_3_COT_START,
]
_COT_END_ALTERNATIVES = [
    _GRANITE_3_3_COT_END,
]

_MODEL_NAME = "Granite 3.3"


def _random_uuid() -> str:
    """:returns: hexadecimal data suitable to use as a unique identifier"""
    return str(uuid.uuid4())


@output_processor(
    _GRANITE_3_3_MODEL_NAME,
    # Huggingface
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_8B_HF,
    # Ollama
    "granite3.3",
    _GRANITE_3_3_2B_OLLAMA,
    _GRANITE_3_3_8B_OLLAMA,
)
class Granite3Point3OutputProcessor(OutputProcessor):
    """
    Output processor for version 3.3 of the main Granite models, all sizes.
    """

    def transform(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        # Downcast to a Granite-specific request type with possible additional fields.
        # This operation also performs additional validation.
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        results = []
        for result in output.results:
            output = result.completion_string
            original_output = output

            # Parse out CoT reasoning
            cot = None
            if inputs.thinking:
                cot_start_span = None
                cot_end_span = None
                for cot_start_str in _COT_START_ALTERNATIVES:
                    if (cot_start_pos := output.find(cot_start_str)) != -1:
                        cot_start_span = (
                            cot_start_pos,
                            cot_start_pos + len(cot_start_str),
                        )
                        break
                for cot_end_str in _COT_END_ALTERNATIVES:
                    if (cot_end_pos := output.find(cot_end_str)) != -1:
                        cot_end_span = (cot_end_pos, cot_end_pos + len(cot_end_str))
                        break

                if (
                    cot_start_span
                    and cot_end_span
                    and cot_end_span[0] > cot_start_span[1]
                ):
                    cot = output[cot_start_span[1] : cot_end_span[0]].strip()
                    output = (
                        output[: cot_start_span[0]] + output[cot_end_span[1] :].strip()
                    )
                    resp_start_span = None
                    resp_end_span = None
                    for resp_start_str in [_GRANITE_3_3_RESP_START]:
                        if (resp_start_pos := output.find(resp_start_str)) != -1:
                            resp_start_span = (
                                resp_start_pos,
                                resp_start_pos + len(resp_start_str),
                            )
                            break
                    for resp_end_str in [_GRANITE_3_3_RESP_END]:
                        if (resp_end_pos := output.find(resp_end_str)) != -1:
                            resp_end_span = (
                                resp_end_pos,
                                resp_end_pos + len(resp_end_str),
                            )
                            break
                    if (
                        resp_start_span
                        and resp_end_span
                        and resp_end_span[0] > resp_start_span[1]
                    ):
                        output = output[resp_start_span[1] : resp_end_span[0]].strip()

            # Parse out tool calls
            tool_calls = []
            if inputs.tools and output.startswith("<tool_call>"):
                # Basic tool call parsing: assume well-formed JSON that adheres to the
                # argument schema specified in the request.
                try:
                    tool_calls = json.loads(output[len("<tool_call>") :])
                    if not isinstance(tool_calls, list):
                        raise TypeError("Model didn't output a list of tool calls")
                    tool_calls = [
                        FunctionCall.model_validate(tool_call_json)
                        for tool_call_json in tool_calls
                    ]
                    for function_call in tool_calls:
                        # Model may decide not to produce IDs
                        if function_call.id is None:
                            function_call.id = _random_uuid()
                    # Output has been turned to tools
                    output = ""
                except (ValueError, TypeError, pydantic.ValidationError):
                    # Parsing failed; flow through
                    pass

            # Parse out citations, documents and hallucinations
            try:
                parsed_output = parse_model_output(output, inputs)
            except Exception as err:
                raise ValueError(
                    "Failed to parse citations, documents and hallucinations "
                    "from model output."
                ) from err

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        citations=parsed_output["citations"],
                        content=(parsed_output["response"]),
                        documents=parsed_output["docs"],
                        hallucinations=parsed_output["hallucinations"],
                        reasoning_content=cot,
                        raw=original_output,
                        stop_reason=result.stop_reason,
                        tool_calls=tool_calls,
                    )
                )
            )

        return ChatCompletionResults(results=results)
