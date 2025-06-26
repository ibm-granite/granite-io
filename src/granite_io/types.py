# SPDX-License-Identifier: Apache-2.0

"""
Common shared types
"""

# Standard
from collections.abc import Mapping
from typing import Any, List, Optional, Union

# Third Party
from typing_extensions import Literal, TypeAlias
import pydantic


class FunctionCall(pydantic.BaseModel):
    id: str | None = None
    name: str

    # This field should adhere to the argument schema from the  associated
    # FunctionDefinition in the generation request that produced it.
    arguments: dict[str, Any] | None


class Hallucination(pydantic.BaseModel):
    """Hallucination data as returned by the model output parser"""

    hallucination_id: str
    risk: str
    reasoning: Optional[str] = None
    response_text: str
    response_begin: int
    response_end: int


class Citation(pydantic.BaseModel):
    """Citation data as returned by the model output parser"""

    citation_id: str
    doc_id: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int


class Document(pydantic.BaseModel):
    """Document data as returned by the model output parser"""

    doc_id: str
    text: str


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
    citations: list[Citation] | None = None
    documents: list[Document] | None = None
    hallucinations: list[Hallucination] | None = None
    stop_reason: str | None = None

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

    model_config = pydantic.ConfigDict(
        # Enforce strict formatting of function schema, unlike the parent JSON object.
        extra="forbid"
    )

    def to_openai_json(self) -> dict:
        """
        :returns: JSON representation of this function, as the Python equivalent of the
            standard JSON used in the OpenAI chat completions API for the `tools`
            argument.
        """
        # The Jinja templates whose behavior we emulate just pass through JSON data.
        return self.model_dump()


class GenerateInputs(pydantic.BaseModel):
    """Common inputs for backends

    Attributes:

        OPTIONAL PARAMS
            prompt: The prompt(s) to generate completions for.
            model: Model name or ID.
            best_of: Generates best_of completions server-side. **Deprecated** on most
            platforms.
            echo: Echo back the prompt in addition to the completion.
            frequency_penalty: Penalize new tokens based on their existing frequency.
            logit_bias: Modify the likelihood of specified tokens.
            logprobs: Include the log probabilities on the most likely tokens.
            max_tokens: The maximum number of tokens to generate in the completion.
            n: How many completions to generate for each prompt.
            presence_penalty: Penalize new tokens based on whether they are in the text.
            stop: Sequences where the API will stop generating further tokens.
            stream: Whether to stream back partial progress.
            stream_options: A dictionary containing options for the streaming response.
            suffix: The suffix that comes after a completion of inserted text.
            temperature: The temperature parameter for controlling randomness of output.
            top_p: The top-p parameter for nucleus sampling.
            user: A unique identifier representing your end-user.
            extra_headers: Additional headers to include in the request.
    """

    prompt: Optional[Union[str, List[Union[str, List[Union[str, List[int]]]]]]] = None
    model: Optional[str] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    logprobs: Optional[Union[int, bool]] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Union[Optional[str], List[str], None] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    extra_headers: Optional[Mapping[str, str]] = None
    extra_body: Optional[Mapping[str, Any]] = {}

    model_config = pydantic.ConfigDict(
        # Pass through arbitrary additional keyword arguments for handling by model- or
        # specific I/O processors.
        arbitrary_types_allowed=True,
        extra="allow",
    )


class ChatCompletionInputs(pydantic.BaseModel):
    """
    Class that represents the lowest-common-denominator inputs to a chat completion
    call.  Individual input/output processors can extend this schema with their own
    additional proprietary fields.
    """

    messages: list[ChatMessage]
    tools: list[FunctionDefinition] = []
    generate_inputs: Optional[GenerateInputs] = None
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

    def with_messages(self, new_messages: list[ChatMessage]) -> "ChatCompletionInputs":
        """
        :param new_messages: Updated list of messages in the conversation

        :returns: a copy of this object with the indicated messages list. Does not
        modify the original object.
        """
        return self.model_copy(update={"messages": new_messages})

    def with_next_message(self, next_message: ChatMessage) -> "ChatCompletionInputs":
        """
        :param next_message: Additional message to add to the conversation

        :returns: a copy of this object with one additional message in the messages
        list. Does not modify the original object.
        """
        new_messages = self.messages.copy()
        new_messages.append(next_message)
        return self.with_messages(new_messages)

    def with_addl_generate_params(
        self, params: Mapping[str, object]
    ) -> "ChatCompletionInputs":
        """
        :param params: Additional parameters to add.

        :returns: a version of this object with one additional message in the messages
        list. Does not modify the original object.
        Leaves in place any generation parameters already present.
        """
        previous_inputs = (
            self.generate_inputs
            if self.generate_inputs is not None
            else GenerateInputs()
        )
        new_inputs = previous_inputs.model_copy(update=params)
        return self.model_copy(update={"generate_inputs": new_inputs})


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

    # tokens for logprobs
    tokens: Optional[list[str]] = None

    # token logprobs
    token_logprobs: Optional[list] = None


class GenerateResults(pydantic.BaseModel):
    results: list[GenerateResult]
