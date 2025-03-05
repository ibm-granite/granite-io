# SPDX-License-Identifier: Apache-2.0

# Standard
import re

# Third Party
from litellm import UnsupportedParamsError
import pytest

# Local
from granite_io.backend.openai import OpenAIBackend
from granite_io.types import GenerateResults


@pytest.mark.vcr
@pytest.mark.block_network
def test_simple(backend_x):
    ret = backend_x.generate("hello")
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.vcr
@pytest.mark.block_network
def test_num_return_sequences_1(backend_x):
    ret = backend_x.generate("hello", num_return_sequences=1)
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.vcr
@pytest.mark.block_network
def test_num_return_sequences_3(backend_x):
    try:
        ret = backend_x.generate("what is up?", num_return_sequences=3)
    except UnsupportedParamsError:
        # Specific exception from LiteLLMBackend
        # xfail because LiteLLM is telling us that ollama does not support
        # num_return_sequences > 1, but we can use LiteLLM with watsonx FTW
        pytest.xfail(
            reason="LiteLLMBackend support for num_return_sequences > 1 varies by provider"
        )

    num_returned = len(ret.results)

    if num_returned == 1 and isinstance(backend_x, OpenAIBackend):
        # ollama with OpenAI will just return 1, other OpenAI backends can return 3
        pytest.xfail(
            reason="OpenAIBackend support for num_return_sequences > 1 varies by provider"
        )

    assert isinstance(ret, GenerateResults)
    assert num_returned == 3


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.flaky(retries=3, delay=5)  # VCR recording flakey
@pytest.mark.parametrize("n", [-1, 0])
def test_num_return_sequences_invalid(backend_x, n):
    with pytest.raises(
        ValueError, match=re.escape(f"Invalid value for num_return_sequences ({n})")
    ):
        _ = backend_x.generate("hello", num_return_sequences=n)
