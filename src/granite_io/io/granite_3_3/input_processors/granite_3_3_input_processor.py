# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# Standard
import datetime
import json

# Third Party
from pydantic_core import PydanticCustomError
import pydantic

# Local
from granite_io.io.base import InputProcessor
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
from granite_io.io.registry import input_processor
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

_TODAYS_DATE_STR = datetime.datetime.now().strftime("%B %d, %Y")


def override_date_for_testing(todays_date_str: str | None):
    """Override the date that methods in this file will use for today's date, in order
    to make test outputs consistent.

    :param todays_date_str: Date string to use for generating prompts until further
     notice, or ``None`` to revert to using the real date.
    """
    global _TODAYS_DATE_STR  # pylint: disable=global-statement
    if todays_date_str is None:
        _TODAYS_DATE_STR = datetime.datetime.now().strftime("%B %d, %Y")
    else:
        _TODAYS_DATE_STR = todays_date_str


# String that comes at the beginning of the system message that a Granite 3.3 model must
# receive at the beginning of the prompt for any completion request that does not
# provide a custom system message.
#
# Note that the original Jinja template tends to choose weird dates from the future for
# the "Today's date" part. Instead of replicating that behavior, we put today's actual
# date in that section of the prompt. This difference probably doesn't matter, since
# none of the supervised fine tuning data exercises knowledge cutoffs.
#
# As an additional wrinkle, we need to use a consistent date when testing, so we use a
# function to recreate this string every time we need it.
def _make_system_message_start():
    return f"""\
Knowledge Cutoff Date: April 2024.
Today's Date: {_TODAYS_DATE_STR}.
You are Granite, developed by IBM."""


# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are both tools and RAG documents in the current request.
_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful assistant with access to the following tools. When a tool is \
required to answer the user's query, respond only with <|tool_call|> followed by a JSON \
list of tools used. If a tool does not exist in the provided list of tools, notify the \
user that you do not have the ability to fulfill the request. \

Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are documents in the current request but there are no tools in the current
# request.
_NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
 Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are tools in the current request but there are no documents in the current
# request.
_TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful assistant with access to the following tools. When a tool is \
required to answer the user's query, respond only with <|tool_call|> followed by a JSON \
list of tools used. If a tool does not exist in the provided list of tools, notify the \
user that you do not have the ability to fulfill the request."""

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `True`.
_NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART = f"""\
 You are a helpful AI assistant.
Respond to every user query in a comprehensive and detailed way. You can write down \
your thoughts and reasoning process before responding. In the thought process, engage \
in a comprehensive cycle of analysis, summarization, exploration, reassessment, \
reflection, backtracing, and iteration to develop well-considered thinking process. \
In the response section, based on various attempts, explorations, and reflections from \
the thoughts section, systematically present the final solution that you deem correct. \
The response should summarize the thought process. Write your thoughts between \
{_GRANITE_3_3_COT_START}{_GRANITE_3_3_COT_END} and write your response between \
{_GRANITE_3_3_RESP_START}{_GRANITE_3_3_RESP_END} \
for each user query."""

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `False`.
_NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART = """\
 You are a helpful AI assistant."""


# String that a Granite 3.3 model must receive immediately after either
# _TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE  (if there are tools) or
# _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools) in the system prompt
# if the "citations" flag is `True` and there are documents.
_DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART = """ \

Use the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a \
document in the search result, e.g <start_of_cite> {document_id: 1}my fact <end_of_cite> \
for a fact from document 1. Afterwards, \
list all the citations with their corresponding documents in an ordered list."""

# String that a Granite 3.3 model must receive immediately after either
# _TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are tools and no citations) or
# _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools or citations) or
# _DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART in the system prompt
# if the "hallucinations" flag is `True` and there are documents.
# Note that a list of zero documents counts as "having documents".
_DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART = """ \

Finally, after the response is written, include a numbered list of sentences from the \
response with a corresponding risk value that are hallucinated and not based in the documents."""


class Document(pydantic.BaseModel):
    text: str

    # Document ID is required in Granite 3.3
    doc_id: str | int


class ControlsRecord(pydantic.BaseModel):
    citations: bool | None = None
    hallucinations: bool | None = None
    length: str | None = None  # Length output control variable
    originality: str | None = None

    @pydantic.field_validator("length", mode="after")
    @classmethod
    def _validate_length(cls, value: str | None) -> str | None:
        if value is None or value == "short" or value == "long":
            return value
        raise PydanticCustomError(
            "length field validator",
            'length ({length}) must be "short" or "long" or None',
            {"length": value},
        )

    @pydantic.field_validator("originality", mode="after")
    @classmethod
    def _validate_originality(cls, value: str | None) -> str | None:
        if value is None or value == "extractive" or value == "abstractive":
            return value
        raise PydanticCustomError(
            "originality field validator",
            'originality ({originality}) must be "extractive" or "abstractive" or None',
            {"originality": value},
        )


class Granite3Point3Inputs(ChatCompletionInputs):
    """
    Class that represents the inputs to a Granite 3.3 model generation call.

    Contains fields for all input functionality supported by the current version of
    Granite.

    This class will gain additional fields as new functionality is added to Granite.
    """

    documents: list[Document] = []
    controls: ControlsRecord | None = None

    thinking: bool = False

    @pydantic.field_validator("messages")
    @classmethod
    def _validate_inputs_messages(cls, messages: list) -> list:
        # Make a copy so the validation code below can mutate the messages list but pass
        # through the original value. The caller also might have a pointer to the list.
        original_messages = messages
        messages = messages.copy()

        # There is no supervised fine tuning data for the case of zero messages.
        # Models are not guaranteed to produce a valid response if there are zero
        # messages.
        if len(messages) == 0:
            raise ValueError(
                "No messages. Model behavior for this case is not defined."
            )

        # The first message, and only the first message, may be the system message.
        first_message_is_system_message = isinstance(messages[0], SystemMessage)
        if first_message_is_system_message:
            messages = messages[1:]
            # If there is a system message, there must be at least one more user or
            # assistant message.
            if len(messages) == 0:
                raise ValueError(
                    "Input contains only a system message. Model behavior for this "
                    "case is not defined."
                )

        # The first message that is not a system message must be
        # either a user or assistant message.
        if not isinstance(messages[0], UserMessage | AssistantMessage):
            if first_message_is_system_message:
                raise ValueError(
                    f"First message after system message must be a user or "
                    f"assistant message. Found type {type(messages[0])}"
                )
            raise ValueError(
                f"First message must be a system, user, or assistant "
                f"Found type {type(messages[0])}"
            )

        # Undocumented constraint: All other messages form a conversation that
        # alternates strictly between user and assistant, possibly with tool calls
        # after an assistant turn and before the next user turn.
        # TODO: Validate this invariant.

        # Pydantic will use the value that this validator returns as the value of the
        # messages field. Undo any changes that we made during validation and return
        # the original value.
        return original_messages


@input_processor(
    _GRANITE_3_3_MODEL_NAME,
    # Huggingface
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_8B_HF,
    # Ollama
    "granite3.3",
    _GRANITE_3_3_2B_OLLAMA,
    _GRANITE_3_3_8B_OLLAMA,
)
class Granite3Point3InputProcessor(InputProcessor):
    """
    Input processor for version 3.3 of the main Granite models, all sizes.
    This input processor is based on the Jinja template from tokenizer_config.json.

    ```
      "chat_template": "{# Alias tools -> available_tools #}\n{%- if tools and not available_tools -%}\n    {%- set available_tools = tools -%}\n{%- endif -%}\n{%- if messages[0]['role'] == 'system' %}\n     {%- set system_message = messages[0]['content'] %}\n     {%- set loop_messages = messages[1:] %}\n {%- else %}\n     {%- set system_message = \" Knowledge Cutoff Date: April 2024.\n Today's Date: \" + strftime_now('%B %d, %Y') + \". You are Granite, developed by IBM.\" %}\n     {%- if available_tools and documents %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request. \nWrite the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n     {%- elif available_tools %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.\" %}\n     {%- elif documents %}\n         {%- set system_message = system_message + \" Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n    {%- elif thinking %}\n    {%- set system_message = system_message + \" You are a helpful AI assistant.\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query.\" %}\n     {%- else %}\n         {%- set system_message = system_message + \" You are a helpful AI assistant.\" %}\n     {%- endif %}\n     {%- if 'citations' in controls and documents %}\n         {%- set system_message = system_message + ' \nUse the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a document in the search result, e.g <|start_of_cite|> {document_id: 1}my fact <|end_of_cite|> for a fact from document 1. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}\n     {%- endif %}\n     {%- if 'hallucinations' in controls and documents %}\n         {%- set system_message = system_message + ' \nFinally, after the response is written, include a numbered list of sentences from the response with a corresponding risk value that are hallucinated and not based in the documents.' %}\n     {%- endif %}\n     {%- set loop_messages = messages %}\n {%- endif %}\n {{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\n' }}\n {%- if available_tools %}\n     {{- '<|start_of_role|>available_tools<|end_of_role|>' }}\n     {{- available_tools | tojson(indent=4) }}\n     {{- '<|end_of_text|>\n' }}\n {%- endif %}\n {%- if documents %}\n     {%- for document in documents %}\n         {{- '<|start_of_role|>document {\"document_id\": \"' + document['doc_id'] | string + '\"}<|end_of_role|>\n' }}\n         {{- document['text'] }}\n         {{- '<|end_of_text|>\n' }}\n              {%- endfor %}\n {%- endif %}\n {%- for message in loop_messages %}\n     {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n     {%- if loop.last and add_generation_prompt %}\n         {{- '<|start_of_role|>assistant' }}\n             {%- if controls %}\n                 {{- ' ' + controls | tojson()}}\n             {%- endif %}\n         {{- '<|end_of_role|>' }}\n     {%- endif %}\n {%- endfor %}",
    ```
    """

    def _split_messages(
        self, inputs: Granite3Point3Inputs
    ) -> tuple[SystemMessage | None, list[UserMessage]]:
        """
        Separate the system message from other messages.

        :returns: Tuple of system message, if present, and remaining messages.
        """
        messages = inputs.messages

        # Validation code in the Inputs class should already have verified that there
        # are either zero or one system messages, and that the system message, if
        # present, occurs at position zero.
        if isinstance(messages[0], SystemMessage):
            # First message is a system message.
            return messages[0], messages[1:]
        return None, messages

    def _build_default_system_message(self, inputs: Granite3Point3Inputs) -> str:
        """
        :param inputs: All inputs to a completion request that does not include a custom
            system message.
        :returns: The standard system message portion of the prompt for the request,
            as a string suitable to feed to the model's tokenizer.
        """
        # Compute the predicates that determine exactly what default system message to
        # use.
        have_documents = inputs.documents is not None and len(inputs.documents) > 0
        have_tools = len(inputs.tools) > 0

        # Carefully hew to the policy that the original Jinja template's behavior
        # defines.
        # First, disallow the cases that the authors of the Jinja template did not
        # provide any code to handle.
        if inputs.thinking and have_documents:
            raise ValueError(
                f"'thinking' flag is set, but documents were provided. "
                f"{_GRANITE_3_3_MODEL_NAME} only supports the 'thinking' flag when "
                f"documents are not provided."
            )
        if inputs.thinking and have_tools:
            raise ValueError(
                f"'thinking' flag is set, but tools were provided. "
                f"{_GRANITE_3_3_MODEL_NAME} only supports the 'thinking' flag when "
                f"tools are not provided."
            )

        # The default system message starts with a header that includes the date and
        # knowledge cutoff.
        system_message = "<|start_of_role|>system<|end_of_role|>"
        system_message += _make_system_message_start()

        # Add a middle part that varies depending on tools, documents, and citations.
        if have_documents and have_tools:
            system_message += _TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_documents:  # and not have_tools
            system_message += _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_tools:  # and not have_documents
            system_message += _TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART
        elif inputs.thinking:  # if not have_documents and not have_tools
            system_message += _NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART
        else:  # if not inputs.thinking and not have_documents and not have_tools
            system_message += _NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART

        # Next comes an optional section of instructions for citations.
        if inputs.controls and inputs.controls.citations:
            if not have_documents:
                # TODO: The template skips the citations instruction in this case.
                # Is this behavior an error? Should we raise an error if the caller
                # sets the citations flag but provides zero documents?
                pass
            else:  # if have_documents
                system_message += _DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART

        # Then comes an optional section of instructions for hallucinations.
        if inputs.controls and inputs.controls.hallucinations:
            if not have_documents:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input does not "
                    f"include documents. {_GRANITE_3_3_MODEL_NAME} only supports the "
                    f"'hallucinations' flag when documents are provided."
                )
            # if have_documents
            system_message += _DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART

        # Finish with an end of text
        system_message += "<|end_of_text|>\n"

        return system_message

    def _message_to_prompt_string(self, message: UserMessage | AssistantMessage) -> str:
        if isinstance(message, UserMessage):
            return (
                f"<|start_of_role|>user<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, AssistantMessage):
            # Note that we discard any tool calls in the message, per the Jinja
            # template.
            return (
                f"<|start_of_role|>assistant<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, ToolResultMessage):
            # Note that we discard the tool call ID, per the Jinja template.
            return (
                f"<|start_of_role|>tool<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        raise TypeError(f"Unexpected message type {type(message)}")

    def _build_controls_record(self, inputs: Granite3Point3Inputs) -> dict | None:
        """
        Use the output control flags in ``inputs`` to build a version of the
        undocumented arbitrary JSON data regarding output controls that the Jinja
        template expected to see in the input for each chat completion request.

        :returns: A fake JSON record for "controls", or nothing of no output control
        flags were set.
        """
        if not inputs.controls:
            return None
        result = {}
        if inputs.controls.citations:
            # The following is a guess; we have no example data for this case.
            result["citations"] = True
        if inputs.controls.hallucinations:
            # The following is a guess; we have no example data for this case.
            result["hallucinations"] = True
        if inputs.controls.length is not None:
            result["length"] = inputs.controls.length
        if inputs.controls.originality is not None:
            result["originality"] = inputs.controls.originality

        if len(result) == 0:
            return None
        return result

    def transform(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        # Downcast to a Granite-specific request type with possible additional fields.
        # This operation also performs additional validation.
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        # Check for a caller-provided system message
        system_message_json, loop_messages = self._split_messages(inputs)

        if system_message_json is not None:
            if inputs.thinking:
                raise ValueError(
                    f"'thinking' flag is set, but the model input includes a custom "
                    f"system message. {_GRANITE_3_3_MODEL_NAME} only supports the "
                    f"'thinking' flag when the default system message is used."
                )
            if len(inputs.documents) > 0:
                raise ValueError(
                    f"The model input includes documents and a custom system message. "
                    f"{_GRANITE_3_3_MODEL_NAME} only supports the documents list when "
                    f"the default system message is used."
                )
            if inputs.controls and inputs.controls.citations:
                raise ValueError(
                    f"'citations' flag is set, but the model input includes a custom "
                    f"system message. {_GRANITE_3_3_MODEL_NAME} only supports the "
                    f"'citations' flag when the default system message is used."
                )
            if inputs.controls and inputs.controls.hallucinations:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input includes a "
                    f"custom system message. {_GRANITE_3_3_MODEL_NAME} only supports "
                    f"the 'hallucinations' flag when the default system message is "
                    f"used."
                )
            system_message = (
                f"<|start_of_role|>system<|end_of_role|>"
                f"{system_message_json.content}<|end_of_text|>\n"
            )
        else:  # if system_message_json is None:
            # No caller-provided system message.
            # Create a default system message according to the rules implied by the
            # tokenizer's Jinja template.
            system_message = self._build_default_system_message(inputs)

        if len(inputs.tools) == 0:
            tools_part = ""
        else:
            tools_part = (
                "<|start_of_role|>available_tools<|end_of_role|>"
                + json.dumps([t.to_openai_json() for t in inputs.tools], indent=4)
                + "<|end_of_text|>\n"
            )

        if len(inputs.documents) == 0:
            documents_part = ""
        else:
            documents_part = "".join(
                [
                    f'<|start_of_role|>document {{"document_id": "{d.doc_id}"}}'
                    f"<|end_of_role|>\n{d.text}<|end_of_text|>\n"
                    for d in inputs.documents
                ]
            )

        messages_part = "".join(
            [self._message_to_prompt_string(message) for message in loop_messages]
        )

        # Jinja template expects arbitrary JSON, while our dataclass has specific
        # fields for supported controls.
        controls_record = self._build_controls_record(inputs)
        controls_str = (
            "" if controls_record is None else " " + json.dumps(controls_record)
        )

        generation_prompt_part = (
            ""
            if not add_generation_prompt
            else f"<|start_of_role|>assistant{controls_str}<|end_of_role|>"
        )

        return (
            system_message
            + tools_part
            + documents_part
            + messages_part
            + generation_prompt_part
        )
