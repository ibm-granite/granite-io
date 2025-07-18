# SPDX-License-Identifier: Apache-2.0

# Standard
import collections.abc

# Third Party
import pytest

# Local
from granite_io import make_backend
from granite_io.backend import Backend
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.consts import (
    _GRANITE_3_2_2B_HF,
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_2B_OLLAMA,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    override_date_for_testing as g32_override_date_for_testing,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    override_date_for_testing as g33_override_date_for_testing,
)
from granite_io.io.rag_agent_lib import obtain_lora


def _no_pings_please(request):
    """pytest request filter that removes ping requests."""
    if "ping" in request.path:
        print(f"Filtering out request {request}")
        return None
    return request


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "before_record_request": _no_pings_please,
        # Use body of POST requests to index into the cassette. Otherwise vcrpy won't
        # detect changes to the prompt.
        "match_on": ["uri", "method", "body"],
        # Regenerate cassette files if deleted; otherwise fail the test if it produces
        # a request that doesn't match the existing cassette.
        "record_mode": "once",
    }


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


def backend_3_3_openai() -> Backend:
    return make_backend(
        "openai",
        {
            "model_name": _GRANITE_3_3_2B_OLLAMA,
            "openai_api_key": "ollama",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )


def backend_3_3_litellm() -> Backend:
    return make_backend(
        "litellm",
        {
            "model_name": "ollama/" + _GRANITE_3_3_2B_OLLAMA,
        },
    )


def backend_3_3_transformers() -> Backend:
    return make_backend(
        "transformers",
        {
            "model_name": _GRANITE_3_3_2B_HF,
        },
    )


@pytest.fixture(scope="function")
def _use_fake_date():
    """
    Granite 3 system prompts include the current date.
    Tests that record network I/O need today's date to be constant so that ``vcrpy``
    can detect whether the other parts of the prompt have changed, invalidating
    recorded outgoing inference requests.

    By wrapping the creation of this date in a fixture, we can be sure to reset to
    normal behavior if a test fails.

    The name of this fixture starts with an underscore so that we don't need to disable
    pylint's unused-argument / W0613 warning on every test that uses this fixture.

    :returns: a fake version of today's date that will be used by Granite 3 input
      processors for the duration of the current test case.
    """
    fake_date = "April 1, 2025"
    g32_override_date_for_testing(fake_date)
    g33_override_date_for_testing(fake_date)
    yield fake_date

    # Cleanup code. Augment as needed as we add new IO processors with date-dependent
    # prompts.
    g32_override_date_for_testing(None)
    g33_override_date_for_testing(None)


@pytest.fixture(
    scope="session", params=[backend_openai, backend_litellm, backend_transformers]
)
def backend_x(request) -> Backend:
    return request.param()


@pytest.fixture(
    scope="session",
    params=[backend_3_3_openai, backend_3_3_litellm, backend_3_3_transformers],
)
def backend_3_3(request) -> Backend:
    return request.param()


@pytest.fixture(scope="session")
def lora_server_session_scoped() -> collections.abc.Generator[
    LocalVLLMServer, object, None
]:
    """
    Session-scoped fixture that runs a local vLLM server.

    The server uses a fixed port because the ``vcrpy`` package requires fixed local
    ports.

    Test cases that use ``vcrpy`` should use the :func:`lora_server()` fixture so that
    they skip actually starting up the server unless they are planning to perform
    network I/O.

    :returns: vLLM server with all the LoRAs for which we currently have IO processors

    """

    # Updated to use Granite 3.3 8B with latest LoRA adapters
    base_model = "ibm-granite/granite-3.3-8b-instruct"

    # LoRA adapter short names - these will be resolved to local paths using
    # obtain_lora()
    lora_adapter_names = [
        "answerability_prediction",  # Maps to answerability_prediction_lora
        "certainty",  # Maps to certainty_lora
        "citation_generation",  # Maps to citation_generation_lora
        "hallucination_detection",  # Maps to hallucination_detection_lora
        "query_rewrite",  # Maps to query_rewrite_lora
        "context_relevancy",  # Maps to context_relevancy_lora
        "prm",  # Maps to prm_lora
    ]

    # Download and get local paths for all LoRA adapters
    lora_adapters = []
    for lora_name in lora_adapter_names:
        try:
            if lora_name == "prm":
                lora_path = "ibm-granite/granite-3.3-8b-lora-math-prm"
            else:
                lora_path = obtain_lora(lora_name)
            lora_adapters.append((lora_name, str(lora_path)))
            print(f"✅ Downloaded LoRA adapter: {lora_name} -> {lora_path}")
        except (OSError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to download LoRA adapter {lora_name}: {e}")
            # Continue with other adapters

    server = LocalVLLMServer(
        base_model, lora_adapters=lora_adapters, port=35782, max_model_len=8192
    )
    # server.wait_for_startup(200)
    yield server

    # Shutdown code runs at end of test session
    server.shutdown()


@pytest.fixture(scope="function")
# pylint: disable-next=redefined-outer-name
def lora_server(lora_server_session_scoped, vcr):
    """
    Wrapper for :func:`lora_server_session_scoped()` that triggers server startup only
    from tests that have ``vcrpy`` recording active.
    """
    if not vcr.write_protected:
        # Recording active; ensure server is running
        lora_server_session_scoped.wait_for_startup(200)
    return lora_server_session_scoped
