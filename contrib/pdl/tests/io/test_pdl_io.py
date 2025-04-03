# SPDX-License-Identifier: Apache-2.0

"""
Simple placeholder for unit testing contrib
"""

# Third Party
from src.io.pdl_io import PlaceholderInputOutputProcessor


def test_pdl_io():
    """Simple example test"""
    ret = PlaceholderInputOutputProcessor()
    assert ret is not None
