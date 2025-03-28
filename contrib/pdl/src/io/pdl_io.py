# SPDX-License-Identifier: Apache-2.0

"""
Prompt Data Language (PDL) I/O processor
"""

# Local
from granite_io.io.base import InputOutputProcessor
from granite_io.optional import import_optional
from granite_io.types import ChatCompletionInputs, ChatCompletionResult


class PlaceholderInputOutputProcessor(InputOutputProcessor):
    """
    Placeholder to be replaced by incoming PDL I/O processor
    """

    with import_optional("pdl"):
        # Third Party
        # pylint: disable-next=import-error,import-outside-toplevel
        from pdl.pdl_ast import FunctionBlock, PdlLocationType

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResult:
        """Dummy placeholder for PDL implementation"""

        _ = inputs  # temporary for lint
        return ChatCompletionResult()
