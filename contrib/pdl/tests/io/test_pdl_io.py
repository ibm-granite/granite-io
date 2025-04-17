# SPDX-License-Identifier: Apache-2.0

"""
Simple placeholder for unit testing contrib
"""

# Third Party
from src.granite_io_pdl.pdl_io import (  # pylint: disable=import-error
    PdlInputOutputProcessor,
)

pdl_processor = """
function:
  inputs: obj
return:
  lastOf:
  - model: "granite3.2:2b"
    backend: openai
    input: ${ inputs.messages }
    modelResponse: results
  - ${ results }
"""


def test_pdl_io():
    """Simple example test"""
    ret = PdlInputOutputProcessor(pdl=pdl_processor)
    assert ret is not None
