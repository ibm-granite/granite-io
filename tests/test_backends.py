# SPDX-License-Identifier: Apache-2.0

# Third Party
from litellm import UnsupportedParamsError
import pytest

# Local
from granite_io.backend.openai import OpenAIBackend
from granite_io.types import GenerateResults


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.block_network
async def test_simple(backend_x):
    ret = await backend_x(prompt="hello")
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.block_network
async def test_num_return_sequences_1(backend_x):
    ret = await backend_x(prompt="hello", num_return_sequences=1)
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.block_network
async def test_num_return_sequences_3(backend_x):
    try:
        ret = await backend_x(prompt="what is up?", num_return_sequences=3)
    except UnsupportedParamsError:
        # Specific exception from LiteLLMBackend
        # xfail because LiteLLM is telling us that ollama does not support
        # num_return_sequences > 1, but we can use LiteLLM with watsonx FTW
        pytest.xfail("LiteLLMBackend support for n > 1 varies by provider")

    num_returned = len(ret.results)

    if num_returned == 1 and isinstance(backend_x, OpenAIBackend):
        # ollama with OpenAI will just return 1, other OpenAI backends can return 3
        pytest.xfail("OpenAIBackend support for n > 1 varies by provider")

    assert isinstance(ret, GenerateResults)
    assert num_returned == 3


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.parametrize("n", [-1, 0])
async def test_num_return_sequences_invalid(backend_x, n):
    with pytest.raises(
        ValueError, match=(f"Invalid value for (n|num_return_sequences) \({n}\)")
    ):
        _ = await backend_x(prompt="hello", num_return_sequences=n)
