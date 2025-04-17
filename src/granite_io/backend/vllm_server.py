# SPDX-License-Identifier: Apache-2.0

# Note: Never rename this file to "vllm.py". You will have a bad time.

# Standard
from collections.abc import Iterable
import asyncio
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


class LocalVLLMServer:
    """
    Class that manages a vLLM server subprocess on the local machine.
    """

    def __init__(
        self,
        model_name: str,
        device_name: str = "auto",
        served_model_name: str | None = None,
        api_key: str | None = None,
        port: int | None = None,
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 32768,
        enforce_eager: bool = True,
        log_level: str = "INFO",
        lora_adapters: Iterable[tuple[str, str]] = tuple(),
        max_lora_rank: int = 64,
    ):
        """
        :param model_name: Path to local file or Hugging Face coordinates of model
        :param served_model_name: Optional alias under which the model should be named
         in the OpenAI API
        :param device_name: Name of vLLM device to use for inference. Current options
         are {auto,cuda,neuron,cpu,tpu,xpu,hpu}.
         Note that the "cpu" option requires building vLLM from source.
        :param api_key: Optional API key for the server to require. Otherwise this class
         will generate a random key.
        :param port: Optional port on localhost to use. If not specified, this class
         will pick a random unused port.
        :param gpu_memory_utilization: What fraction of the GPU's memory to dedicate
         to the target model
        :param max_model_len: Maximum context length to use before truncating to avoid
         running out of GPU memory.
        :param enforce_eager: If ``True`` skip compilation to make the server start up
         faster.
        :param log_level: Logging level for the vLLM subprocess
        :param lora_adapters: Map from model name to LoRA adapter location
        :param max_lora_rank: vLLM needs you to specify an upper bound on the size of
         the shared dimension of the low-rank approximations in LoRA adapters.
        """
        self._model_name = model_name
        self._served_model_name = served_model_name
        self._lora_adapters = lora_adapters
        self._lora_names = [t[0] for t in lora_adapters]

        vllm_exec = shutil.which("vllm")
        if vllm_exec is None:
            raise ValueError("vLLM not installed.")

        if not port:
            # Find an open port on localhost
            with socketserver.TCPServer(("localhost", 0), None) as s:
                port = s.server_address[1]
        self._server_port = port

        # Generate shared secret so that other local processes can't hijack our server.
        self._api_key = api_key if api_key else str(uuid.uuid4())

        environment = os.environ.copy()
        # Disable annoying log messages about current throughput being zero
        # Unfortunately the only documented way to do this is to turn off all
        # logging.
        # TODO: Look for undocumented solutions.
        environment["VLLM_LOGGING_LEVEL"] = log_level
        environment["VLLM_API_KEY"] = self._api_key

        # Immediately start up a server process on the open port
        command_parts = [
            vllm_exec,
            "serve",
            model_name,
            "--port",
            str(self._server_port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--guided_decoding_backend",
            "outlines",
            "--device",
            device_name,
        ]
        if enforce_eager:
            command_parts.append("--enforce-eager")
        if served_model_name is not None:
            command_parts.append("--served-model-name")
            command_parts.append(served_model_name)
        if len(lora_adapters) > 0:
            command_parts.append("--enable-lora")
            command_parts.append("--max_lora_rank")
            command_parts.append(str(max_lora_rank))
            command_parts.append("--lora-modules")
            for k, v in lora_adapters:
                command_parts.append(f"{k}={v}")

        logger.info("Running: %s", " ".join(command_parts))  # pylint: disable=logging-not-lazy
        self._subproc = subprocess.Popen(command_parts, env=environment)  # pylint: disable=consider-using-with

    def __repr__(self):
        return f"LocalVLLMServer({self._model_name} -> {self._base_url()})"

    def _base_url(self):
        return f"http://localhost:{self._server_port}"

    @property
    def openai_url(self) -> str:
        return f"{self._base_url()}/v1"

    @property
    def openai_api_key(self) -> str:
        return self._api_key

    def wait_for_startup(self, timeout_sec: float | None = None):
        """
        Blocks  until the server has started.
        :param timeout_sec: Optional upper limit for how long to block. If this
         limit is reached, this method will raise a TimeoutError
        """
        start_sec = time.time()
        while timeout_sec is None or time.time() - start_sec < timeout_sec:
            try:  # Exceptions as control flow due to library design
                with urllib.request.urlopen(self._base_url() + "/ping") as response:
                    _ = response.read().decode("utf-8")
                return  # Success
            except urllib.error.URLError:
                time.sleep(1)
        raise TimeoutError(
            f"Failed to connect to {self._base_url()} after {timeout_sec} seconds."
        )

    async def await_for_startup(self, timeout_sec: float | None = None):
        """
        Blocks the local coroutine until the server has started.
        :param timeout_sec: Optional upper limit for how long to block. If this
         limit is reached, this method will raise a TimeoutError
        """
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
        self._subproc.send_signal(signal.SIGINT)

    def make_backend(self) -> OpenAIBackend:
        """
        :returns: A backend instance pointed at the primary model that our subprocess
         is serving.
        """
        return OpenAIBackend(
            aconfig.Config(
                {
                    "model_name": (
                        self._served_model_name
                        if self._served_model_name
                        else self._model_name
                    ),
                    "openai_base_url": f"{self._base_url()}/v1",
                    "openai_api_key": self._api_key,
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
                    "openai_api_key": self._api_key,
                }
            )
        )
