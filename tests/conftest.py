# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# Local
from granite_io import make_backend
from granite_io.backend import Backend
from granite_io.io.granite_3_2 import GRANITE_3_2_2B_HF


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
            "model_name": GRANITE_3_2_2B_HF,
        },
    )


@pytest.fixture(
    scope="session", params=[backend_openai, backend_litellm, backend_transformers]
)
def backend_x(request) -> Backend:
    return request.param()
