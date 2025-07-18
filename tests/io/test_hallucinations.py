# SPDX-License-Identifier: Apache-2.0

"""
Test cases for hallucination.py
"""

# Standard
import datetime

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.base import RewriteRequestProcessor
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.hallucinations import (
    HallucinationsCompositeIOProcessor,
    HallucinationsIOProcessor,
)
from granite_io.types import (
    GenerateResult,
    GenerateResults,
)

_LORA_NAME = "hallucination_detection"

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the visibility level of Git Repos and Issue "
                "Tracking projects?",
            },
            {
                "role": "assistant",
                "content": "Git Repos and Issue Tracking projects can have one of "
                "three visibility levels: "
                "private, internal, or public. Private projects are visible only to "
                "project members, internal "
                "projects are visible to all logged-in IBM Cloud users, and public "
                "projects are visible to anyone.",
            },
        ],
        "documents": [
            {
                "doc_id": 1,
                # Original text
                "text": "Git Repos and Issue Tracking is an IBM-hosted component of \
the Continuous Delivery service. All of the data that you provide to Git Repos and \
Issue Tracking, including but not limited to source files, issues, pull requests, and \
project configuration properties, is managed securely within Continuous Delivery. \
However, Git Repos and Issue Tracking supports various mechanisms for exporting, \
sending, or otherwise sharing data to users and third parties. The ability of Git \
Repos and Issue Tracking to share information is typical of many social coding \
platforms. However, such sharing might conflict with regulatory controls that \
apply to your business. After you create a project in Git Repos and Issue Tracking, \
but before you entrust any files, issues, records, or other data with the project, \
review the project settings and change any settings that you deem necessary to \
protect your data. Settings to review include visibility levels, email notifications, \
integrations, web hooks, access tokens, deploy tokens, and deploy keys. Project \
visibility levels \n\nGit Repos and Issue Tracking projects can have one of the \
following visibility levels: private, internal, or public. * Private projects are \
visible only to project members. This setting is the default visibility level for new \
projects, and is the most secure visibility level for your data. * Internal projects \
are visible to all users that are logged in to IBM Cloud. * Public projects are \
visible to anyone. To limit project access to only project members, complete the \
following steps:\n\n\n\n1. From the project sidebar, click Settings > General. \
2. On the General Settings page, click Visibility > project features > permissions. \
3. Locate the Project visibility setting. 4. Select Private, if it is not already \
selected. 5. Click Save changes. Project membership \n\nGit Repos and Issue Tracking \
is a cloud hosted social coding environment that is available to all Continuous \
Delivery users. If you are a Git Repos and Issue Tracking project Maintainer or Owner, \
you can invite any user and group members to the project. IBM Cloud places no \
restrictions on who you can invite to a project.",
            },
            {
                "doc_id": 2,
                "text": "After you create a project in Git Repos and Issue Tracking, \
but before you entrust any files, issues, records, or other data with the project, \
review the project settings and change any settings that are necessary to protect your \
data. \
Settings to review include visibility levels, email notifications, integrations, web \
hooks, access tokens, deploy tokens, and deploy keys. Project visibility levels \
\n\nGit Repos and Issue Tracking projects can have one of the following visibility \
levels: private, internal, or public. * Private projects are visible only to \
project members. This setting is the default visibility level for new projects, and \
is the most secure visibility level for your data. * Internal projects are visible to \
all users that are logged in to IBM Cloud. * Public projects are visible to anyone. \
To limit project access to only project members, complete the following \
steps:\n\n\n\n1. From the project sidebar, click Settings > General. 2. On the \
General Settings page, click Visibility > project features > permissions. 3. Locate \
the Project visibility setting. 4. Select Private, if it is not already selected. \
5. Click Save changes. Project email settings \n\nBy default, Git Repos and Issue \
Tracking notifies project members by way of email about project activities. These \
emails typically include customer-owned data that was provided to Git Repos and Issue \
Tracking by users. For example, if a user posts a comment to an issue, Git Repos and \
Issue Tracking sends an email to all subscribers. The email includes information such \
as a copy of the comment, the user who posted it, and when the comment was posted. \
To turn off all email notifications for your project, complete the following \
steps:\n\n\n\n1. From the project sidebar, click Settings > General. 2. On the \
**General Settings **page, click Visibility > project features > permissions. \
3. Select the Disable email notifications checkbox. 4. Click Save changes. Project \
integrations and webhooks",
            },
        ],
    }
)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
    )


_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")


def test_canned_input():
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    io_processor = HallucinationsIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT)
    print("#####")
    print(output.prompt)
    print("#####")
    expected_prompt = f"""\
<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: {_TODAYS_DATE}.
You are Granite, developed by IBM. Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.<|end_of_text|>
<|start_of_role|>document {{"document_id": "1"}}<|end_of_role|>
Git Repos and Issue Tracking is an IBM-hosted component of the Continuous Delivery service. All of the data that you provide to Git Repos and Issue Tracking, including but not limited to source files, issues, pull requests, and project configuration properties, is managed securely within Continuous Delivery. However, Git Repos and Issue Tracking supports various mechanisms for exporting, sending, or otherwise sharing data to users and third parties. The ability of Git Repos and Issue Tracking to share information is typical of many social coding platforms. However, such sharing might conflict with regulatory controls that apply to your business. After you create a project in Git Repos and Issue Tracking, but before you entrust any files, issues, records, or other data with the project, review the project settings and change any settings that you deem necessary to protect your data. Settings to review include visibility levels, email notifications, integrations, web hooks, access tokens, deploy tokens, and deploy keys. Project visibility levels 

Git Repos and Issue Tracking projects can have one of the following visibility levels: private, internal, or public. * Private projects are visible only to project members. This setting is the default visibility level for new projects, and is the most secure visibility level for your data. * Internal projects are visible to all users that are logged in to IBM Cloud. * Public projects are visible to anyone. To limit project access to only project members, complete the following steps:



1. From the project sidebar, click Settings > General. 2. On the General Settings page, click Visibility > project features > permissions. 3. Locate the Project visibility setting. 4. Select Private, if it is not already selected. 5. Click Save changes. Project membership 

Git Repos and Issue Tracking is a cloud hosted social coding environment that is available to all Continuous Delivery users. If you are a Git Repos and Issue Tracking project Maintainer or Owner, you can invite any user and group members to the project. IBM Cloud places no restrictions on who you can invite to a project.<|end_of_text|>
<|start_of_role|>document {{"document_id": "2"}}<|end_of_role|>
After you create a project in Git Repos and Issue Tracking, but before you entrust any files, issues, records, or other data with the project, review the project settings and change any settings that are necessary to protect your data. Settings to review include visibility levels, email notifications, integrations, web hooks, access tokens, deploy tokens, and deploy keys. Project visibility levels 

Git Repos and Issue Tracking projects can have one of the following visibility levels: private, internal, or public. * Private projects are visible only to project members. This setting is the default visibility level for new projects, and is the most secure visibility level for your data. * Internal projects are visible to all users that are logged in to IBM Cloud. * Public projects are visible to anyone. To limit project access to only project members, complete the following steps:



1. From the project sidebar, click Settings > General. 2. On the General Settings page, click Visibility > project features > permissions. 3. Locate the Project visibility setting. 4. Select Private, if it is not already selected. 5. Click Save changes. Project email settings 

By default, Git Repos and Issue Tracking notifies project members by way of email about project activities. These emails typically include customer-owned data that was provided to Git Repos and Issue Tracking by users. For example, if a user posts a comment to an issue, Git Repos and Issue Tracking sends an email to all subscribers. The email includes information such as a copy of the comment, the user who posted it, and when the comment was posted. To turn off all email notifications for your project, complete the following steps:



1. From the project sidebar, click Settings > General. 2. On the **General Settings **page, click Visibility > project features > permissions. 3. Select the Disable email notifications checkbox. 4. Click Save changes. Project integrations and webhooks<|end_of_text|>
<|start_of_role|>user<|end_of_role|>What is the visibility level of Git Repos and Issue Tracking projects?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|><i0> Git Repos and Issue Tracking projects can have one of three visibility levels: private, internal, or public. <i1> Private projects are visible only to project members, internal projects are visible to all logged-in IBM Cloud users, and public projects are visible to anyone.<|end_of_text|>
<|start_of_role|>system<|end_of_role|>Split the last assistant response into individual sentences. For each sentence in the last assistant response, identify the faithfulness by comparing with the provided documents and generate the faithfulness reasoning and faithfulness decision. Ensure that your output includes all response sentence IDs, and for each response sentence ID, provide the corresponding faithfulness reasoning and faithfulness decision. The output must be a json structure.<|end_of_text|>\
"""  # noqa: E501
    assert output.prompt == expected_prompt


def test_canned_output():
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = HallucinationsIOProcessor(None)
    raw_output_to_expected = [
        (
            '[{"i": 0, "r": "This sentence makes a factual claim about the visibility '
            "levels of Git Repos and Issue Tracking projects. The document states "
            "'Git Repos and Issue Tracking projects can have one of the following "
            "visibility levels: private, internal, or public.' This matches exactly "
            'with the claim in the sentence.", "f": "faithful"}, {"i": 1, "r": "This '
            "sentence makes factual claims about the visibility of each type of "
            "project. The document states 'Private projects are visible only to "
            "project members,' 'Internal projects are visible to all users that "
            "are logged in to IBM Cloud,' and 'Public projects are visible to "
            "anyone.' These statements match exactly with the claims in the "
            'sentence.", "f": "faithful"}]',
            [
                {
                    "hallucination_id": "0",
                    "risk": "faithful",
                    "reasoning": "This sentence makes a factual claim about the "
                    "visibility levels of Git Repos and Issue Tracking projects. "
                    "The document states 'Git Repos and Issue Tracking projects can "
                    "have one of the following visibility levels: private, internal, "
                    "or public.' This matches exactly with the claim in the sentence.",
                    "response_text": "Git Repos and Issue Tracking projects can have "
                    "one of three visibility levels: private, internal, or public.",
                    "response_begin": 0,
                    "response_end": 108,
                },
                {
                    "hallucination_id": "1",
                    "risk": "faithful",
                    "reasoning": "This sentence makes factual claims about the "
                    "visibility of each type of project. The document states 'Private "
                    "projects are visible only to project members,' 'Internal projects "
                    "are visible to all users that are logged in to IBM Cloud,' and '"
                    "Public projects are visible to anyone.' These statements match "
                    "exactly with the claims in the sentence.",
                    "response_text": "Private projects are visible only to project "
                    "members, internal projects are visible to all logged-in IBM Cloud "
                    "users, and public projects are visible to anyone.",
                    "response_begin": 109,
                    "response_end": 269,
                },
            ],
        ),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), _EXAMPLE_CHAT_INPUT
        )
        assert len(output.results) == 1
        result_json = [
            h.model_dump() for h in output.results[0].next_message.hallucinations
        ]
        print("!!!!!!")
        print(result_json)
        print("!!!!!!")
        assert result_json == expected


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    lora_backend = lora_server.make_lora_backend(_LORA_NAME)
    io_proc = HallucinationsIOProcessor(lora_backend)

    # Pass our example input thorugh the I/O processor and retrieve the result
    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    assert chat_result.results[0].next_message.hallucinations is not None


@pytest.mark.vcr
def test_run_composite(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the composite I/O processor.
    """
    granite_backend = lora_server.make_backend()
    lora_backend = lora_server.make_lora_backend(_LORA_NAME)
    granite_io_proc = make_io_processor("Granite 3.3", backend=granite_backend)
    io_proc = HallucinationsCompositeIOProcessor(granite_io_proc, lora_backend)

    # Strip off last message and rerun
    input_without_msg = _EXAMPLE_CHAT_INPUT.model_copy(
        update={"messages": _EXAMPLE_CHAT_INPUT.messages[:-1]}
    ).with_addl_generate_params({"temperature": 0.2, "n": 5})
    results = io_proc.create_chat_completion(input_without_msg)
    assert len(results.results) == 5


@pytest.mark.vcr
def test_run_processor(lora_server: LocalVLLMServer, _use_fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the composite I/O processor.
    """
    lora_backend = lora_server.make_lora_backend(_LORA_NAME)
    io_proc = HallucinationsIOProcessor(lora_backend)
    request_proc = RewriteRequestProcessor(io_proc)

    results = request_proc.process(_EXAMPLE_CHAT_INPUT)
    assert len(results) == 1
