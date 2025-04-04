# SPDX-License-Identifier: Apache-2.0

# Standard
import collections.abc

# Third Party
import pytest
import torch

# Local
from granite_io import make_backend
from granite_io.backend import Backend
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.consts import _GRANITE_3_2_2B_HF


@pytest.fixture(scope="session")
def vcr_config():
    return {"filter_headers": ["authorization"]}


def backend_openai() -> Backend:
    return make_backend(
        "openai",
        {
            "model_name": "granite3.2:2b",
            "openai_api_key": "ollama",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )


def backend_litellm() -> Backend:
    return make_backend(
        "litellm",
        {
            "model_name": "ollama/" + "granite3.2:2b",
        },
    )


def backend_transformers() -> Backend:
    return make_backend(
        "transformers",
        {
            "model_name": _GRANITE_3_2_2B_HF,
        },
    )


@pytest.fixture(
    scope="session", params=[backend_openai, backend_litellm, backend_transformers]
)
def backend_x(request) -> Backend:
    return request.param()


@pytest.fixture(scope="session")
def lora_server() -> collections.abc.Generator[LocalVLLMServer, object, None]:
    """
    :returns: vLLM server with all the LoRAs for which we currently have IO processors
    """
    if not torch.cuda.is_available():
        pytest.xfail("GPU required to run vLLM. vLLM required to run model.")

    # Currently all adapters are trained on Granite 3.2 8B
    base_model = "ibm-granite/granite-3.2-8b-instruct"
    lora_adapters = [
        # Format is server name, model name
        ("certainty", "ibm-granite/granite-uncertainty-3.2-8b-lora"),
    ]

    server = LocalVLLMServer(base_model, lora_adapters=lora_adapters)
    server.wait_for_startup(200)
    yield server

    # Shutdown code runs at end of test session
    server.shutdown()
