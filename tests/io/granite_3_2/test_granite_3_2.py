# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# pylint: disable=redefined-outer-name,broad-exception-caught

# Standard
import json

# Third Party
from litellm import UnsupportedParamsError
from pydantic import ValidationError
import pytest
import transformers

# Local
from granite_io import make_io_processor
from granite_io.backend import Backend
from granite_io.backend.litellm import LiteLLMBackend
from granite_io.backend.openai import OpenAIBackend
from granite_io.backend.transformers import TransformersBackend
from granite_io.io.consts import (
    _GRANITE_3_2_COT_END,
    _GRANITE_3_2_COT_START,
)
from granite_io.io.granite_3_2.granite_3_2 import (
    _MODEL_NAME,
    GRANITE_3_2_2B_HF,
    Granite3Point2InputOutputProcessor,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    _ControlsRecord,
    _Granite3Point2Inputs,
)
from granite_io.io.granite_3_2.output_processors.granite_3_2_output_parser import (
    _CITATION_START,
    _HALLUCINATION_START,
)
from granite_io.io.granite_3_2.output_processors.granite_3_2_output_processor import (
    _COT_END_ALTERNATIVES,
    _COT_START_ALTERNATIVES,
)
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResults,
    Citation,
    Document,
    GenerateInputs,
    GenerateResult,
    GenerateResults,
    Hallucination,
    UserMessage,
)

## Helpers #####################################################################

# All the different chat completion requests that are tested in this file, serialized as
# JSON strings. Represented as a dictionary instead of a list so that pytest output will
# show the short key instead of the long value when referencing a single run of a test
INPUT_JSON_STRS = {
    "simple": """
{
    "messages":
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"}
    ]
}
""",
    "thinking_tag": """
{
    "messages":
    [
        {"role": "user", "content": "How much wood could a wood chuck chuck?"}
    ],
    "thinking": true
}
""",
    "stop_strings": """
{
    "messages":
        [
            {"role": "user", "content": "How much wood could a wood chuck chuck?"}
        ],
    "generate_inputs":
        {
            "stop": "woodchuck"
        }
}
""",
}


@pytest.fixture(scope="session", params=INPUT_JSON_STRS)
def input_json_str(request: pytest.FixtureRequest) -> str:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different chat completion requests."""
    return INPUT_JSON_STRS[request.param]


@pytest.fixture(scope="session")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    model_path = GRANITE_3_2_2B_HF
    try:
        ret = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=False
        )
    except Exception as e:
        pytest.skip(f"No tokenizer for {model_path}: {e}")
    return ret


msg = UserMessage(content="Hello")
no_thinking_input = ChatCompletionInputs(messages=[msg])
thinking_input = ChatCompletionInputs(messages=[msg], thinking=True)

thought = "Think think"
response = "respond respond"
pre_thought = "something before"
no_cot_output = f"{thought} {response}"
no_thinking_output = f"{thought} {_GRANITE_3_2_COT_END} {response}"
no_response_output = f"{_GRANITE_3_2_COT_START}\n\n{response}"
cot_output = (
    f"{_GRANITE_3_2_COT_START}\n\n{thought}\n{_GRANITE_3_2_COT_END}\n\n{response}"
)
cot_alt_output = f"{_COT_START_ALTERNATIVES[-1]}\n\n{thought}\n{_COT_END_ALTERNATIVES[-1]}\n\n{response}"
cot_mixed_output = (
    f"{_GRANITE_3_2_COT_START}\n\n{thought}\n{_COT_END_ALTERNATIVES[-1]}\n\n{response}"
)
cot_pre_output = f"{pre_thought} {_GRANITE_3_2_COT_START} {thought} {_COT_END_ALTERNATIVES[-1]} {response}"

no_constituent_output = "Mad about dog!"
citation_example = '<co>1</co> Document 0: "Dog info"'
citation_output = (
    f"{no_constituent_output}<co>1</co>\n\n{_CITATION_START}\n\n{citation_example}\n\n"
)
hallucination_example = "1. Risk low: Mad about dog"
citation_hallucination_output = (
    f"{citation_output}{_HALLUCINATION_START}\n\n{hallucination_example}\n\n"
)
expected_citation = Citation(
    citation_id="1",
    doc_id="0",
    context_text="Dog info",
    context_begin=0,
    context_end=8,
    response_text="Mad about dog!",
    response_begin=0,
    response_end=14,
)
expected_document = Document(doc_id="0", text="Dog info")
doc_input = ChatCompletionInputs(
    messages=[msg], documents=[{"doc_id": "0", "text": "Dog info"}]
)
expected_hallucination = Hallucination(
    hallucination_id="1",
    risk="low",
    response_text="Mad about dog",
    response_begin=0,
    response_end=13,
)
## Tests #######################################################################


@pytest.mark.parametrize(
    ["length", "originality", "error"],
    [
        (None, None, None),
        ("short", None, None),
        (None, "abstractive", None),
        ("long", "extractive", None),
        ("BAD_VAL", "abstractive", "input_value='BAD_VAL'"),
        ("long", "BAD_VAL", "input_value='BAD_VAL'"),
        ("BAD_VAL", "Another Bad Value", "input_value='BAD_VAL'"),
        ("ShOrT", None, "input_value='ShOrT'"),
        (None, "aBsTrAcTiVe", "input_value='aBsTrAcTiVe'"),
        (1, None, "input_type=int"),
        (None, 2, "input_type=int"),
    ],
)
def test_controls_field_validators(length, originality, error):
    if error:
        with pytest.raises(ValidationError, match=error):
            _ControlsRecord(length=length, originality=originality)
    else:
        _ControlsRecord(length=length, originality=originality)


def test_read_inputs(input_json_str):
    """
    Verify that the dataclasses for the Granite 3.2 I/O processor can parse Granite 3.2 JSON
    """
    print(f"{input_json_str=}")
    input_json = json.loads(input_json_str)
    input_obj = ChatCompletionInputs.model_validate(input_json)
    input_obj_2 = ChatCompletionInputs.model_validate_json(input_json_str)

    assert input_obj == input_obj_2

    # Parse additional Granite-specific fields
    granite_input_obj = _Granite3Point2Inputs.model_validate(input_obj.model_dump())

    # Verify that we can convert back to JSON without crashing
    granite_input_obj.model_dump_json()
    input_obj.model_dump_json()


def test_same_input_string(
    tokenizer: transformers.PreTrainedTokenizerBase, input_json_str: str
):
    """
    Verify that the I/O processor produces the exact same input string as the Jinja
    template that ships with the model.
    """

    # First apply the Jinja template
    input_json = json.loads(input_json_str)
    input_kwargs = input_json.copy()
    del input_kwargs["messages"]
    transformers_str = tokenizer.apply_chat_template(
        input_json["messages"],
        **input_kwargs,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Then compare against the input processor
    inputs = ChatCompletionInputs.model_validate_json(input_json_str)
    io_proc_str = Granite3Point2InputOutputProcessor().inputs_to_string(inputs)

    assert io_proc_str == transformers_str


def test_basic_inputs_to_string():
    """
    Basic test against canned output in case the developer doesn't have a way to load
    an actual Granite 3.2 tokenizer for output comparisons.

    Chat input:

    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    Expected similar (dates will vary) chat template request generated:

    <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
    Today's Date: February 17, 2025.
    You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Hello, how are you?<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>I'm doing great. How can I help you today?<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>I'd like to show off how chat templating works!<|end_of_text|>
    """
    chatRequest = Granite3Point2InputOutputProcessor().inputs_to_string(
        inputs=ChatCompletionInputs(
            messages=[
                UserMessage(content="Hello, how are you?"),
                AssistantMessage(content="I'm doing great. How can I help you today?"),
                UserMessage(content="I'd like to show off how chat templating works!"),
            ]
        ),
        add_generation_prompt=False,
    )
    print(f"Chat request: {chatRequest}")

    chatReqStart = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date:"
    assert chatRequest.startswith(chatReqStart)

    chatReqModelMsg = "You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>"
    assert chatReqModelMsg in chatRequest

    chatReqBody = """\
<|start_of_role|>user<|end_of_role|>Hello, how are you?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>I'm doing great. How can I help you today?<|end_of_text|>
<|start_of_role|>user<|end_of_role|>I'd like to show off how chat templating works!<|end_of_text|>"""
    assert chatReqBody in chatRequest

    assert chatRequest.endswith("")


@pytest.mark.vcr
def test_completion_repetition_param(backend_x: Backend):
    messages = [
        {
            "role": "user",
            "content": "Can you answer my question?",
        }
    ]

    # What a client might be sending to a backend
    generate_inputs = {
        "prompt": "Just give me an example of what you can do",
        "temperature": 0.5,
        "repetition_penalty": 0.5,
    }
    inputs = ChatCompletionInputs(messages=messages, generate_inputs=generate_inputs)

    io_processor = make_io_processor(_MODEL_NAME, backend=backend_x)
    try:
        outputs: ChatCompletionResults = io_processor.create_chat_completion(inputs)
    except TypeError as te:
        if isinstance(backend_x, OpenAIBackend):
            pytest.xfail(str(te))
        raise te
    except UserWarning as uw:
        if isinstance(backend_x, TransformersBackend):
            pytest.xfail(str(uw))
        raise uw

    assert isinstance(outputs, ChatCompletionResults)


@pytest.mark.vcr
def test_completion_presence_param(backend_x: Backend):
    messages = [
        {
            "role": "user",
            "content": "Can you answer my question?",
        }
    ]
    generate_inputs = {
        "prompt": "Just give me an example of what you can do",
        "temperature": 0.5,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
    }
    inputs = ChatCompletionInputs(messages=messages, generate_inputs=generate_inputs)

    io_processor = make_io_processor(_MODEL_NAME, backend=backend_x)
    try:
        outputs: ChatCompletionResults = io_processor.create_chat_completion(inputs)
    except UnsupportedParamsError as upe:
        # Specific exception from LiteLLMBackend
        pytest.xfail(upe.message)
    except UserWarning as uw:
        if isinstance(backend_x, TransformersBackend):
            pytest.xfail(str(uw))
        raise uw

    assert isinstance(outputs, ChatCompletionResults)


@pytest.mark.vcr
def test_run_processor(backend_x: Backend, input_json_str: str):
    inputs = ChatCompletionInputs.model_validate_json(input_json_str)
    io_processor = make_io_processor(_MODEL_NAME, backend=backend_x)
    outputs: ChatCompletionResults = io_processor.create_chat_completion(inputs)

    assert isinstance(outputs, ChatCompletionResults)
    assert len(outputs.results) == 1

    content = outputs.results[0].next_message.content
    assert content  # Make sure we don't get empty result (I had a bug)

    # Test for stop reason
    if inputs.generate_inputs and inputs.generate_inputs.stop:
        stop = inputs.generate_inputs.stop
        # Note: currently transformers includes the stop tokens,
        # but OpenAI does not. So, either endswith or not-in.
        assert stop not in content
        assert outputs.results[0].next_message.stop_reason == "stop"

    # TODO: Verify outputs in greater detail


@pytest.mark.parametrize(
    ["inputs", "output", "exp_thought", "exp_resp"],
    [
        # No thinking flag
        (no_thinking_input, no_thinking_output, None, no_thinking_output),
        (no_thinking_input, cot_output, None, cot_output),
        # Thinking flag
        (thinking_input, no_cot_output, None, no_cot_output),
        (thinking_input, no_thinking_output, None, no_thinking_output),
        (thinking_input, no_response_output, None, no_response_output),
        (thinking_input, cot_output, thought, response),
        (thinking_input, cot_alt_output, thought, response),
        (thinking_input, cot_mixed_output, thought, response),
        (thinking_input, cot_pre_output, thought, f"{pre_thought} {response}"),
    ],
)
def test_cot_parsing(inputs, output, exp_thought, exp_resp):
    """Test the parsing logic for CoT reasoning output"""
    proc = Granite3Point2InputOutputProcessor()
    generated = GenerateResults(
        results=[
            GenerateResult(
                completion_string=output, completion_tokens=[], stop_reason="?"
            )
        ]
    )
    result = proc.output_to_result(generated, inputs).results[0].next_message
    assert result.reasoning_content == exp_thought
    assert result.content == exp_resp
    assert result.raw == output


@pytest.mark.parametrize(
    [
        "inputs",
        "output",
        "exp_document",
        "exp_citation",
        "exp_hallucination",
        "exp_resp",
    ],
    [
        # No constituents
        (
            no_thinking_input,
            no_constituent_output,
            None,
            None,
            None,
            no_constituent_output,
        ),
        # Citation
        (
            doc_input,
            citation_output,
            [expected_document],
            [expected_citation],
            None,
            no_constituent_output,
        ),
        # Citation and hallucination
        (
            doc_input,
            citation_hallucination_output,
            [expected_document],
            [expected_citation],
            [expected_hallucination],
            no_constituent_output,
        ),
    ],
)
def test_citation_hallucination_parsing(
    inputs, output, exp_document, exp_citation, exp_hallucination, exp_resp
):
    """Test the parsing logic for Rag and hallucinations output"""
    proc = Granite3Point2InputOutputProcessor()
    generated = GenerateResults(
        results=[
            GenerateResult(
                completion_string=output, completion_tokens=[], stop_reason="?"
            )
        ]
    )
    result = proc.output_to_result(generated, inputs).results[0].next_message
    assert result.content == exp_resp
    assert result.citations == exp_citation
    assert result.documents == exp_document
    assert result.hallucinations == exp_hallucination


@pytest.mark.vcr(record_mode="new_episodes")
@pytest.mark.block_network
def test_multiple_return(backend_x: Backend, input_json_str: str):
    inputs = ChatCompletionInputs.model_validate_json(input_json_str)
    inputs = inputs.model_copy(
        update={"generate_inputs": GenerateInputs(max_tokens=1024, n=3)}
    )
    io_processor = make_io_processor(_MODEL_NAME, backend=backend_x)
    try:
        results: ChatCompletionResults = io_processor.create_chat_completion(inputs)
    except UnsupportedParamsError:
        if isinstance(backend_x, LiteLLMBackend):
            pytest.xfail("LiteLLMBackend support for n > 1 varies by provider")

    assert isinstance(results, ChatCompletionResults)
    assert len(results.results) == 3
    # TODO: Verify outputs in greater detail
