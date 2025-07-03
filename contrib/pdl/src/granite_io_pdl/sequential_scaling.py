# SPDX-License-Identifier: Apache-2.0

"""
Sequential Scaling I/O processor
"""

# Standard
from typing import Callable
import pathlib

# Local
from .pdl_io import PdlInputOutputProcessor
from granite_io.io.base import InputOutputProcessor
from granite_io.types import ChatCompletionResults


class SequentialScalingInputOutputProcessor(PdlInputOutputProcessor):
    """
    Input-output processor asking multiple answers until a predicate is satisfied.
    """

    def __init__(
        self,
        generator: InputOutputProcessor,
        validator: Callable[[ChatCompletionResults], bool],
        max_iterations: int = 5,
    ):
        """
        :param generator: Sub-processor over which this processor should sample
        :param validator: predicate that the response must satisfy.
        :param max_iterations: Maximal number of model calls.
        """
        cwd = pathlib.Path(__file__).parent.resolve()
        super().__init__(
            pdl_file=cwd / "sequential_scaling.pdl",
            pdl_scope={
                "generator": generator,
                "validator": validator,
                "k": max_iterations,
            },
        )
