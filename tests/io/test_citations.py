# SPDX-License-Identifier: Apache-2.0

"""
Test cases for io_adapters/citations.py
"""

# Standard
import datetime

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.citations import CitationsCompositeIOProcessor, CitationsIOProcessor
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
    override_date_for_testing,
)
from granite_io.types import GenerateResult, GenerateResults

_EXAMPLE_CHAT_INPUT = Granite3Point2Inputs.model_validate(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the visibility level of Git Repos and Issue \
Tracking projects?",
            },
            {
                "role": "assistant",
                "content": "Git Repos and Issue Tracking projects can have one of the \
following visibility levels: private, internal, or public. Private projects are \
visible only to project members, internal projects are visible to all users that are \
logged in to IBM Cloud, and public projects are visible to anyone. By default, \
new projects are set to private visibility level, which is the most secure for your \
data.",
            },
        ],
        "documents": [
            {
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
restrictions on who you can invite to a project."
            },
            {
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
integrations and webhooks"
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
    io_processor = CitationsIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT)
    print("#####")
    print(output.prompt)
    print("#####")
    expected_prompt = f"""\
<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: {_TODAYS_DATE}.
You are Granite, developed by IBM.Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.<|end_of_text|>
<|start_of_role|>documents<|end_of_role|>Document 0
<c0> Git Repos and Issue Tracking is an IBM-hosted component of the Continuous Delivery service. <c1> All of the data that you provide to Git Repos and Issue Tracking, including but not limited to source files, issues, pull requests, and project configuration properties, is managed securely within Continuous Delivery. <c2> However, Git Repos and Issue Tracking supports various mechanisms for exporting, sending, or otherwise sharing data to users and third parties. <c3> The ability of Git Repos and Issue Tracking to share information is typical of many social coding platforms. <c4> However, such sharing might conflict with regulatory controls that apply to your business. <c5> After you create a project in Git Repos and Issue Tracking, but before you entrust any files, issues, records, or other data with the project, review the project settings and change any settings that you deem necessary to protect your data. <c6> Settings to review include visibility levels, email notifications, integrations, web hooks, access tokens, deploy tokens, and deploy keys. <c7> Project visibility levels 

Git Repos and Issue Tracking projects can have one of the following visibility levels: private, internal, or public. <c8> * Private projects are visible only to project members. <c9> This setting is the default visibility level for new projects, and is the most secure visibility level for your data. <c10> * Internal projects are visible to all users that are logged in to IBM Cloud. <c11> * Public projects are visible to anyone. <c12> To limit project access to only project members, complete the following steps:



1. <c13> From the project sidebar, click Settings > General. <c14> 2. <c15> On the General Settings page, click Visibility > project features > permissions. <c16> 3. <c17> Locate the Project visibility setting. <c18> 4. <c19> Select Private, if it is not already selected. <c20> 5. <c21> Click Save changes. <c22> Project membership 

Git Repos and Issue Tracking is a cloud hosted social coding environment that is available to all Continuous Delivery users. <c23> If you are a Git Repos and Issue Tracking project Maintainer or Owner, you can invite any user and group members to the project. <c24> IBM Cloud places no restrictions on who you can invite to a project.

Document 1
<c25> After you create a project in Git Repos and Issue Tracking, but before you entrust any files, issues, records, or other data with the project, review the project settings and change any settings that are necessary to protect your data. <c26> Settings to review include visibility levels, email notifications, integrations, web hooks, access tokens, deploy tokens, and deploy keys. <c27> Project visibility levels 

Git Repos and Issue Tracking projects can have one of the following visibility levels: private, internal, or public. <c28> * Private projects are visible only to project members. <c29> This setting is the default visibility level for new projects, and is the most secure visibility level for your data. <c30> * Internal projects are visible to all users that are logged in to IBM Cloud. <c31> * Public projects are visible to anyone. <c32> To limit project access to only project members, complete the following steps:



1. <c33> From the project sidebar, click Settings > General. <c34> 2. <c35> On the General Settings page, click Visibility > project features > permissions. <c36> 3. <c37> Locate the Project visibility setting. <c38> 4. <c39> Select Private, if it is not already selected. <c40> 5. <c41> Click Save changes. <c42> Project email settings 

By default, Git Repos and Issue Tracking notifies project members by way of email about project activities. <c43> These emails typically include customer-owned data that was provided to Git Repos and Issue Tracking by users. <c44> For example, if a user posts a comment to an issue, Git Repos and Issue Tracking sends an email to all subscribers. <c45> The email includes information such as a copy of the comment, the user who posted it, and when the comment was posted. <c46> To turn off all email notifications for your project, complete the following steps:



1. <c47> From the project sidebar, click Settings > General. <c48> 2. <c49> On the **General Settings **page, click Visibility > project features > permissions. <c50> 3. <c51> Select the Disable email notifications checkbox. <c52> 4. <c53> Click Save changes. <c54> Project integrations and webhooks<|end_of_text|>
<|start_of_role|>user<|end_of_role|>What is the visibility level of Git Repos and Issue Tracking projects?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|><r0> Git Repos and Issue Tracking projects can have one of the following visibility levels: private, internal, or public. <r1> Private projects are visible only to project members, internal projects are visible to all users that are logged in to IBM Cloud, and public projects are visible to anyone. <r2> By default, new projects are set to private visibility level, which is the most secure for your data.<|end_of_text|>
<|start_of_role|>system<|end_of_role|>Split the last assistant response into individual sentences. For each sentence in the response, identify the statement IDs from the documents that it references. Ensure that your output includes all response sentence IDs, and for each response sentence ID, provide the corresponding referring document sentence IDs.<|end_of_text|>\
"""  # noqa: E501
    assert output.prompt == expected_prompt


def test_canned_output():
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = CitationsIOProcessor(None)

    # Map raw input to just the citation offsets
    raw_output_to_expected = [
        (
            '{"<r0>": ["<c7>"], "<r1>": ["<c8>", "<c10>", "<c11>"]}',
            [
                {
                    "context_begin": 1034,
                    "context_end": 1178,
                    "response_begin": 0,
                    "response_end": 116,
                },
                {
                    "context_begin": 1179,
                    "context_end": 1234,
                    "response_begin": 117,
                    "response_end": 289,
                },
                {
                    "context_begin": 1353,
                    "context_end": 1430,
                    "response_begin": 117,
                    "response_end": 289,
                },
                {
                    "context_begin": 1431,
                    "context_end": 1471,
                    "response_begin": 117,
                    "response_end": 289,
                },
            ],
        ),
        ('{"<r0>": [], "<r1>": []}', []),
        # IO processor currently raises an exception for debugging in this case.
        # ("<invalid model response>", "ERROR"),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), _EXAMPLE_CHAT_INPUT
        )
        assert len(output.results) == 1
        # Pull out just the citation offsets for comparison
        citation_offsets = [
            {
                key: getattr(c, key)
                for key in (
                    "context_begin",
                    "context_end",
                    "response_begin",
                    "response_end",
                )
            }
            for c in output.results[0].next_message.citations
        ]
        assert citation_offsets == expected


@pytest.mark.vcr
def test_run_model(lora_server: LocalVLLMServer, fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the I/O processor.
    """
    backend = lora_server.make_lora_backend("citations")
    io_proc = CitationsIOProcessor(backend)
    override_date_for_testing(fake_date)  # For consistent VCR output

    # Pass our example input thorugh the I/O processor and retrieve the result

    chat_result = io_proc.create_chat_completion(_EXAMPLE_CHAT_INPUT)

    # Results vary slightly due to rounding error inside the model.
    assert 4 <= len(chat_result.results[0].next_message.citations) <= 5


@pytest.mark.vcr
def test_run_composite(lora_server: LocalVLLMServer, fake_date: str):
    """
    Run a chat completion through the LoRA adapter using the composite I/O processor.
    """
    granite_backend = lora_server.make_backend()
    lora_backend = lora_server.make_lora_backend("citations")
    granite_io_proc = make_io_processor("Granite 3.2", backend=granite_backend)
    io_proc = CitationsCompositeIOProcessor(granite_io_proc, lora_backend)
    override_date_for_testing(fake_date)  # For consistent VCR output

    # Strip off last message and rerun
    input_without_msg = _EXAMPLE_CHAT_INPUT.model_copy(
        update={"messages": _EXAMPLE_CHAT_INPUT.messages[:-1]}
    ).with_addl_generate_params({"temperature": 0.2, "n": 5})
    results = io_proc.create_chat_completion(input_without_msg)
    assert len(results.results) == 5
