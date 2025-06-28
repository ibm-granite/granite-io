# SPDX-License-Identifier: Apache-2.0

# Note: Never rename this file to "vllm.py". You will have a bad time.

# Standard
from collections.abc import Iterable
import asyncio
import dataclasses
import logging
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import time
import urllib
import uuid

# Third Party
import aconfig
import aiohttp

# Local
from granite_io.backend.openai import OpenAIBackend

# Perform the "set sensible defaults for Python logging" ritual.
logger = logging.getLogger("granite_io.backend.vllm_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(handler)


@dataclasses.dataclass
# pylint: disable-next=too-many-instance-attributes
class VLLMServerConfig:
    """
    Wrapper for parameters to pass to a vLLM server.

    This class implements a subset of the functionality of
    :class:`vllm.config.VllmConfig`. We avoid depending on that vLLM dataclass because
    our library needs to function when vLLM is not installed. In particular, test cases
    need to be able to use :class:`LocalVLLMServer` as a stub on systems where vLLM
    cannot easily be installed due to lack of a GPU.
    """

    ##################################################################################
    # The fields that follow mirror vllm.config.ModelConfig

    model: str
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""

    max_model_len: int | str | None = None
    """Model context length (prompt and output). If unspecified, will be
    automatically derived from the model config.

    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format. Examples:\n
    - 1k -> 1000\n
    - 1K -> 1024\n
    - 25.6k -> 25,600"""

    enforce_eager: bool = False
    """Whether to always use eager-mode PyTorch. If True, we will disable CUDA
    graph and always execute the model in eager mode. If False, we will use
    CUDA graph and eager execution in hybrid for maximal performance and
    flexibility."""

    served_model_name: str | list[str] | None = None
    """The model name(s) used in the API. If multiple names are provided, the
    server will respond to any of the provided names. The model name in the
    model field of a response will be the first name in this list. If not
    specified, the model name will be the same as the `--model` argument. Noted
    that this name(s) will also be used in `model_name` tag content of
    prometheus metrics, if multiple names provided, metrics tag will take the
    first one."""

    ##################################################################################
    # The fields that follow are in vllm.config.CacheConfig:
    gpu_memory_utilization: float = 0.9
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""

    ##################################################################################
    # The fields that follow are in vllm.config.LoRAConfig:

    max_lora_rank: int = 16
    """Max LoRA rank."""

    ##################################################################################
    # The fields that follow control VLLM-specific environment variables.
    api_key: str | None = None
    """Single (!) API key for the server's OpenAI-compatible API. Corresponds to the
    environment variable ``VLLM_API_KEY``"""

    logging_level: str = "INFO"
    """Name of the highest log level that vLLM should log at. Corresponds to the 
    environment variable ``VLLM_LOGGING_LEVEL``"""

    ##################################################################################
    # The fields that follow control command-line flags that don't have a dataclass
    # field
    port: int | None = None
    """Port on which the OpenAI-compatible REST API will listen. Controls the
    ``--port`` command-line argument."""

    lora_adapters: Iterable[tuple[str, str]] = tuple()
    """
    List of name, model coordinates pairs for any LoRA adapters to load. Controls the
    ``--lora-modules`` command-line argument."""


class LocalVLLMServer:
    """
    Class that manages a vLLM server subprocess on the local machine.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int = 32768,
        enforce_eager: bool = True,
        served_model_name: str | list[str] | None = None,
        gpu_memory_utilization: float = 0.45,
        max_lora_rank: int = 64,
        api_key: str | None = None,
        logging_level: str = "INFO",
        port: int | None = None,
        lora_adapters: Iterable[tuple[str, str]] = tuple(),
    ):
        """
        :param model_name: Path to local file or Hugging Face coordinates of model
        :param max_model_len: Maximum context length to use before truncating to avoid
         running out of GPU memory.
        :param enforce_eager: If ``True`` skip compilation to make the server start up
         faster.
        :param served_model_name: Optional alias under which the model should be named
         in the OpenAI API. Can be a list of multiple names.
        :param gpu_memory_utilization: What fraction of the GPU's memory to dedicate
         to the target model
        :param max_lora_rank: vLLM needs you to specify an upper bound on the size of
         the shared dimension of the low-rank approximations in LoRA adapters.
        :param api_key: Optional API key for the server to require. Otherwise this class
         will generate a random key.
        :param logging_level: Logging level for the vLLM subprocess
        :param port: Optional port on localhost to use. If not specified, this class
         will pick a random unused port.
        :param lora_adapters: Map from model name to LoRA adapter location
        """
        if served_model_name is None:
            self._primary_model_name = model_name
        elif isinstance(served_model_name, list):
            self._primary_model_name = served_model_name[0]
        elif isinstance(served_model_name, str):
            self._primary_model_name = served_model_name
        else:
            raise TypeError(
                f"Invalid type {type(served_model_name)} for "
                f"served_model_name parameter"
            )
        self._lora_names = [t[0] for t in lora_adapters]

        if not port:
            # Find an open port on localhost
            with socketserver.TCPServer(("localhost", 0), None) as s:
                port = s.server_address[1]

        # Generate shared secret so that other local processes can't hijack our server.
        api_key = api_key if api_key else str(uuid.uuid4())

        # Defer server startup so that test cases can use this class as a stub with
        # vcrpy.
        self._vllm_server_config = VLLMServerConfig(
            model_name,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            served_model_name=served_model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_lora_rank=max_lora_rank,
            api_key=api_key,
            logging_level=logging_level,
            port=port,
            lora_adapters=lora_adapters,
        )
        self._subproc = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self._vllm_server_config.model} "
            f"-> {self._base_url()})"
        )

    def _base_url(self):
        return f"http://localhost:{self._vllm_server_config.port}"

    @property
    def openai_url(self) -> str:
        return f"{self._base_url()}/v1"

    @property
    def openai_api_key(self) -> str:
        return self._api_key

    def start_subprocess(self):
        """
        Trigger startup of the vLLM subprocess.
        Does nothing if the subprocess is already running.
        """
        if self._subproc is not None:
            return

        config = self._vllm_server_config

        vllm_exec = shutil.which("vllm")
        if vllm_exec is None:
            raise ValueError("vLLM not installed.")

        environment = os.environ.copy()
        # Disable annoying log messages about current throughput being zero
        # Unfortunately the only documented way to do this is to turn off all
        # logging.
        # TODO: Look for undocumented solutions.
        environment["VLLM_LOGGING_LEVEL"] = config.logging_level
        environment["VLLM_API_KEY"] = config.api_key

        # Immediately start up a server process on the open port
        command_parts = [
            vllm_exec,
            "serve",
            config.model,
            "--port",
            str(config.port),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-model-len",
            str(config.max_model_len),
            # As of June 2025, there is no value for the guided_decoding_backend
            # argument that works across the three most recent major releases of vLLM
            # "--guided_decoding_backend",
            # "auto",
        ]
        if config.enforce_eager:
            command_parts.append("--enforce-eager")
        if config.served_model_name is not None:
            command_parts.append("--served-model-name")
            if isinstance(config.served_model_name, list):
                command_parts.append(",".join(config.served_model_name))
            else:
                command_parts.append(config.served_model_name)
        if len(config.lora_adapters) > 0:
            command_parts.append("--enable-lora")
            command_parts.append("--max_lora_rank")
            command_parts.append(str(config.max_lora_rank))
            command_parts.append("--lora-modules")
            for k, v in config.lora_adapters:
                command_parts.append(f"{k}={v}")

        logger.info("Running: %s", " ".join(command_parts))  # pylint: disable=logging-not-lazy
        self._subproc = subprocess.Popen(command_parts, env=environment)  # pylint: disable=consider-using-with

    def wait_for_startup(self, timeout_sec: float | None = None):
        """
        Blocks  until the server has started.

        Triggers server startup if the caller has not already called
        :func:`start_subprocess()`.
        :param timeout_sec: Optional upper limit for how long to block. If this
         limit is reached, this method will raise a TimeoutError
        """
        self.start_subprocess()  # Idempotent
        start_sec = time.time()
        while timeout_sec is None or time.time() - start_sec < timeout_sec:
            try:  # Exceptions as control flow due to library design
                with urllib.request.urlopen(self._base_url() + "/ping") as response:
                    _ = response.read().decode("utf-8")
                return  # Success
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        raise TimeoutError(
            f"Failed to connect to {self._base_url()} after {timeout_sec} seconds."
        )

    async def await_for_startup(self, timeout_sec: float | None = None):
        """
        Blocks the local coroutine until the server has started.

        Triggers server startup if the caller has not already called
        :func:`start_subprocess()`.
        :param timeout_sec: Optional upper limit for how long to block. If this
         limit is reached, this method will raise a TimeoutError
        """
        self.start_subprocess()  # Idempotent
        start_sec = time.time()
        while timeout_sec is None or time.time() - start_sec < timeout_sec:
            try:  # Exceptions as control flow due to aiohttp library design
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(self._base_url() + "/ping") as resp,
                ):
                    await resp.text()
                return  # Success
            except (ConnectionRefusedError, aiohttp.ClientConnectorError):
                await asyncio.sleep(1)
        raise TimeoutError(
            f"Failed to connect to {self._base_url()} after {timeout_sec} seconds."
        )

    def shutdown(self):
        # Sending SIGINT to the vLLM process seems to be the only way to stop it.
        # DO NOT USE SIGKILL!!!
        if self._subproc is not None:
            self._subproc.send_signal(signal.SIGINT)
        self._subproc = None

    def make_backend(self) -> OpenAIBackend:
        """
        :returns: A backend instance pointed at the primary model that our subprocess
         is serving.
        """
        return OpenAIBackend(
            aconfig.Config(
                {
                    "model_name": (self._primary_model_name),
                    "openai_base_url": f"{self._base_url()}/v1",
                    "openai_api_key": self._vllm_server_config.api_key,
                }
            )
        )

    def make_lora_backend(self, lora_name: str) -> OpenAIBackend:
        """
        :param lora_name: Name of one of the LoRA adapters that was passed to the
        constructor of this object.

        :returns: A backend instance pointed at the specified LoRA adapter.
        """
        if lora_name not in self._lora_names:
            raise ValueError(
                f"Unexpected LoRA adapter name {lora_name}. Known names "
                f"are: {self._lora_names}"
            )
        return OpenAIBackend(
            aconfig.Config(
                {
                    "model_name": lora_name,
                    "openai_base_url": f"{self._base_url()}/v1",
                    "openai_api_key": self._vllm_server_config.api_key,
                }
            )
        )
