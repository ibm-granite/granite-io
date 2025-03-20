# SPDX-License-Identifier: Apache-2.0

__doc__ = f"""
The {__package__} library provides a common programming abstraction for using
the advanced control instructions with Granite models.
"""

# Local
from granite_io.backend import backend, make_backend  # noqa: F401
from granite_io.io import (  # noqa: F401
    get_input_processor,
    get_io_processor,
    get_output_processor,
    input_processor,
    io_processor,
    output_processor,
)
from granite_io.io.base import make_new_io_processor  # noqa: F401
