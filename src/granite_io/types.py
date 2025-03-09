# SPDX-License-Identifier: Apache-2.0

"""
Common shared types
"""

# Standard
from typing import Any

# Third Party
from typing_extensions import Literal, TypeAlias
import pydantic


class FunctionCall(pydantic.BaseModel):
    id: str
    name: str

    # This field should adhere to the argument schema from the  associated
    # FunctionDefinition in the generation request that produced it.
    arguments: dict[str, Any] | None


class _ChatMessageBase(pydantic.BaseModel):
    """Base class for all message types.

    Due to the vaguaries of Pydantic's JSON parser, we use this class only for common
    functionality, and NOT for defining a common dataclass base type. Use the
    :class:`ChatMessage` type alias to annotate a field or argument as accepting all
    subclasses of this one."""

    content: str
    """Every message has raw string content, even if it also contains parsed structured
    content such as a JSON record."""

    def to_openai_json(self):
        result = {"role": self.role, "content": self.content}
        return result


class UserMessage(_ChatMessageBase):
    role: Literal["user"] = "user"


class AssistantMessage(_ChatMessageBase):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[FunctionCall] = []
    reasoning_content: str | None = None
    # Raw response content without any parsing for re-serialization
    _raw: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw = kwargs.pop("raw", None)

    @property
    def raw(self) -> str:
        """Get the raw content of the response"""
        return self._raw if self._raw is not None else self.content


class ToolResultMessage(_ChatMessageBase):
    role: Literal["tool"] = "tool"
    tool_call_id: str


class SystemMessage(_ChatMessageBase):
    role: Literal["system"] = "system"


ChatMessage: TypeAlias = (
    UserMessage | AssistantMessage | ToolResultMessage | SystemMessage
)
"""Type alias for all message types. We use this Union instead of the actual base class
:class:`_ChatMessageBase` so that Pydantic can parse the message list from JSON."""


class FunctionDefinition(pydantic.BaseModel):
    name: str
    description: str | None = None

    # This field holds a JSON schema for a record, but the `jsonschema` package doesn't
    # define an object type for such a schema, instead using a dictionary.
    parameters: dict[str, Any] | None = None

    def to_openai_json(self) -> dict:
        """
        :returns: JSON representation of this function, as the Python equivalent of the
            standard JSON used in the OpenAI chat completions API for the `tools`
            argument.
        """
        raise NotImplementedError("TODO: Implement this")


class ChatCompletionInputs(pydantic.BaseModel):
    """
    Class that represents the lowest-common-denominator inputs to a chat completion
    call.  Individual input/output processors can extend this schema with their own
    additional proprietary fields.
    """

    messages: list[ChatMessage]
    tools: list[FunctionDefinition] = []
    num_return_sequences: int = 1

    model_config = pydantic.ConfigDict(
        # Pass through arbitrary additional keyword arguments for handling by model- or
        # specific I/O processors.
        extra="allow"
    )

    def __getattr__(self, name: str) -> any:
        """Allow attribute access for unknown attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None

    def with_next_message(self, next_message: ChatMessage) -> "ChatCompletionInputs":
        """Convenience method to create a version of this object with one additional
        message in the messages list."""
        new_messages = self.messages.copy()
        new_messages.append(next_message)
        return self.model_copy(update={"messages": new_messages})


class ChatCompletionResult(pydantic.BaseModel):
    """
    Class that represents the lowest-common-denominator outputs of a chat completion
    call.  Individual input/output processors can extend this schema with their own
    additional proprietary fields.
    """

    next_message: ChatMessage

    def __getattr__(self, name: str) -> any:
        """Allow attribute access for unknown attributes"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None


class ChatCompletionResults(pydantic.BaseModel):
    results: list[ChatCompletionResult]


class GenerateResult(pydantic.BaseModel):
    """
    All the things that our internal :func:`generate()` methods return,
    rolled into a dataclass for ease of maintenance.
    """

    # Not including input characters
    completion_string: str

    # Not including input tokens
    completion_tokens: list[int]

    stop_reason: str


class GenerateResults(pydantic.BaseModel):
    results: list[GenerateResult]
