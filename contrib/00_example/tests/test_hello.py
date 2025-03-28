# SPDX-License-Identifier: Apache-2.0

"""
Simple placeholder for unit testing contrib
"""

# Third Party
from src.hello_lib import hello_world


def test_hello():
    """Simple example test"""
    ret = hello_world()
    assert ret == "Hello today!"  # pylint: disable=comparison-of-constants

    # pylint: disable-next=comparison-of-constants, comparison-with-itself
    assert "Hello today!" == "Hello today!"
