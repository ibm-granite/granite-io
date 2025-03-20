# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import re

# Third Party
from litellm import UnsupportedParamsError
import pytest

# Local
from granite_io import make_backend
from granite_io.backend.openai import OpenAIBackend
from granite_io.types import GenerateInputs, GenerateResults


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_simple(backend_x):
    ret = await backend_x(GenerateInputs(prompt="hello"))
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_num_return_sequences_1(backend_x):
    ret = await backend_x(GenerateInputs(prompt="hello", n=1))
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_num_return_sequences_3(backend_x):
    try:
        ret = await backend_x(GenerateInputs(prompt="what is up?", n=3))
    except UnsupportedParamsError:
        # Specific exception from LiteLLMBackend
        # xfail because LiteLLM is telling us that ollama does not support
        # n > 1, but we can use LiteLLM with watsonx FTW
        pytest.xfail("LiteLLMBackend support for n > 1 varies by provider")

    num_returned = len(ret.results)

    if num_returned == 1 and isinstance(backend_x, OpenAIBackend):
        # ollama with OpenAI will just return 1, other OpenAI backends can return 3
        pytest.xfail("OpenAIBackend support for n > 1 varies by provider")

    assert isinstance(ret, GenerateResults)
    assert num_returned == 3


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Needs OpenAI API key")
async def test_extra_headers():
    """Test extra_headers using manual test case against a server to verify.

    Use environment variables suitable for an OpenAI backend:
    OPENAI_BASE_URI - base URL to the provider
    MODEL_NAME - model name override for the provider
    OPENAI_API_KEY - api key used for the provider
    Note:  For this specific test, we have a provider that needs
           the API key to also be in an additional header.
           It also passes the num_returned test (Ollama does not).
    """
    be = make_backend(
        "openai",
        {
            "model_name": "ibm-granite/granite-8b-instruct-preview-4k",
            "openai_api_key": "<your api key>",  # Use env var to set
            "openai_base_url": "<your base url>",
        },
    )

    extra_headers = {"RITS_API_KEY": os.environ.get("OPENAI_API_KEY")}
    ret = await be(
        GenerateInputs(prompt="what is up?", n=3, extra_headers=extra_headers)
    )
    num_returned = len(ret.results)
    assert isinstance(ret, GenerateResults)
    assert num_returned == 3  # Note: This would fail if you just hit Ollama


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("n", [-1, 0])
async def test_num_return_sequences_invalid(backend_x, n):
    with pytest.raises(ValueError, match=re.escape(f"Invalid value for n ({n})")):
        _ = await backend_x(inputs=GenerateInputs(prompt="hello", n=n))
