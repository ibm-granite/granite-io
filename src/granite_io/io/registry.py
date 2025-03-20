# SPDX-License-Identifier: Apache-2.0

"""
Registry of all Input, Output & IO Processors
"""

# Local
from granite_io.factory import ImportableFactory

_IO_FACTORY = ImportableFactory("io")
io_processor = _IO_FACTORY.decorator
make_io_processor = _IO_FACTORY.construct

_INPUT_FACTORY = ImportableFactory("input")
input_processor = _INPUT_FACTORY.decorator
get_input_processor = _INPUT_FACTORY.construct

_OUTPUT_FACTORY = ImportableFactory("output")
output_processor = _OUTPUT_FACTORY.decorator
get_output_processor = _OUTPUT_FACTORY.construct
