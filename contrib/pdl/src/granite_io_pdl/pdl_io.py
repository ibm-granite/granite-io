# SPDX-License-Identifier: Apache-2.0

"""
Prompt Description Language (PDL) I/O processor
"""

# Standard
from typing import Any

# Third Party
import aconfig

# Local
from granite_io.io.base import InputOutputProcessor
from granite_io.optional import import_optional
from granite_io.types import ChatCompletionInputs, ChatCompletionResults


class PdlInputOutputProcessor(InputOutputProcessor):
    """
    Base class for input-output processors that work by executing a
    Prompt Declaration Language (PDL) function processing `ChatCompletionInputs`.
    """

    with import_optional("pdl"):
        # Third Party
        # pylint: disable-next=import-error,import-outside-toplevel
        from pdl.pdl_ast import FunctionBlock, PdlLocationType

    _pdl_function: FunctionBlock | None
    """PDL function used to process the input."""
    _pdl_scope: dict[str, Any] | None
    """Initial scope in which the PDL program is executed"""
    _pdl_loc: PdlLocationType
    """Location information in the source code of the PDL function."""

    def __init__(
        self,
        config: aconfig.Config | None = None,
        pdl=None,
        pdl_file: str | None = None,
        pdl_scope: dict[str, Any] | None = None,
    ):
        """
        :param config: Setup config for this IO processor
        :param pdl: PDL function processing the `ChatCompletionInputs`
        :param pdl_file: Name of the PDL file containing the function PDL function
        :param pdl_scope: Initial environment in which the function is executed
        """
        with import_optional("pdl"):
            # Third Party
            from pdl.pdl import parse_file, parse_str
        super().__init__(config)
        if isinstance(pdl, str):
            prog, loc = parse_str(pdl, file_name=pdl_file)
            self._pdl_function = prog.root
            self._pdl_loc = loc
        elif pdl is None and pdl_file is not None:
            prog, loc = parse_file(pdl_file)
            self._pdl_function = prog.root
            self._pdl_loc = loc
        else:
            self._pdl_function = pdl
            self._pdl_loc = None
        self._pdl_scope = pdl_scope

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        return self.create_chat_completion(inputs)

    def create_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        with import_optional("pdl"):
            # Third Party
            from pdl.pdl import exec_program
            from pdl.pdl_ast import CallBlock, Program
        prog = Program(
            CallBlock(
                defs={"_pdl_function": self._pdl_function},
                call="${_pdl_function}",
                args={"inputs": inputs.model_dump()},
            )
        )
        results = exec_program(prog, scope=self._pdl_scope)
        return ChatCompletionResults.model_validate(results)
