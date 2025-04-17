# SPDX-License-Identifier: Apache-2.0

"""
Sequential Scaling I/O processor
"""

# Standard
from typing import Callable
import pathlib

# Third Party
import aconfig

# Local
from .pdl_io import PdlInputOutputProcessor
from granite_io.types import ChatCompletionResults


class SequentialScalingInputOutputProcessor(PdlInputOutputProcessor):
    """
    Input-output processor asking multiple answers until a predicate is satisfied.
    """

    def __init__(
        self,
        config: aconfig.Config | None = None,
        model: str | None = None,
        backend: str | None = None,
        max_iterations: int = 5,
        validator: Callable[[ChatCompletionResults], bool] | None = None,
    ):
        """
        :param config: Setup config for this IO processor
        :param model: Model name used by the backend.
        :param backend: Backend name.
        :param max_iterations: Maximal number of model calls.
        :param validator: predicate that the response must satisfy.
        """
        cwd = pathlib.Path(__file__).parent.resolve()
        super().__init__(
            config,
            pdl_file=cwd / "sequential_scaling.pdl",
            pdl_scope={
                "model": model,
                "backend": backend,
                "k": max_iterations,
                "validator": validator,
            },
        )
