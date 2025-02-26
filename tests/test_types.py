# SPDX-License-Identifier: Apache-2.0

"""
Tests for the core types
"""

# Third Party
import pytest

# Local
from granite_io import types


def test_function_call():
    fc = types.FunctionCall(id="1", name="test_func", arguments={"arg1": "value"})
    assert fc.id == "1"
    assert fc.name == "test_func"
    assert fc.arguments == {"arg1": "value"}


def test_chat_message_types():
    um = types.UserMessage(content="user content")
    am = types.AssistantMessage(content="assistant content", tool_calls=[])
    trm = types.ToolResultMessage(content="tool result content", tool_call_id="123")
    sm = types.SystemMessage(content="system content")

    assert um.role == "user"
    assert am.role == "assistant"
    assert trm.role == "tool"
    assert sm.role == "system"
    assert um.to_openai_json()["role"] == "user"
    assert am.to_openai_json()["role"] == "assistant"
    assert trm.to_openai_json()["role"] == "tool"
    assert sm.to_openai_json()["role"] == "system"


def test_function_definition():
    fd = types.FunctionDefinition(name="test_func", description="Test function")
    assert fd.name == "test_func"
    assert fd.description == "Test function"
    assert fd.parameters is None

    with pytest.raises(NotImplementedError):
        fd.to_openai_json()


def test_chat_completion_inputs():
    cci = types.ChatCompletionInputs(messages=[])
    assert len(cci.messages) == 0
    assert len(cci.tools) == 0

    # Setting additional attributes should work as expected
    cci.additional_attr = "value"
    assert hasattr(cci, "additional_attr") is True
    assert cci.additional_attr == "value"

    # Getting unknown attributes should return None
    assert cci.foobar is None


def test_chat_completion_result():
    ccrr = types.ChatCompletionResult(
        next_message=types.UserMessage(content="test message")
    )
    assert hasattr(ccrr, "next_message") is True

    # Testing __getattr__ behavior for unknown attributes
    assert ccrr.foobar is None
    assert ccrr.bazbat is None


def test_generate_result():
    gr = types.GenerateResult(
        completion_string="generated string",
        completion_tokens=[1, 2, 3],
        stop_reason="stop reason",
    )
    assert gr.completion_string == "generated string"
    assert gr.completion_tokens == [1, 2, 3]
    assert gr.stop_reason == "stop reason"
