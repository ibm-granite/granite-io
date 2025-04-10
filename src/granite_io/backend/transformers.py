# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any
import asyncio
import concurrent.futures
import dataclasses

# Third Party
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.optional import import_optional
from granite_io.types import GenerateInputs, GenerateResult, GenerateResults

if TYPE_CHECKING:
    # Third Party
    import transformers


def _detect_hw_accel():
    """:returns: Pytorch's name for the best available hardware accelerator on the
    current machine."""
    with import_optional("transformers"):
        # Third Party
        import torch
    if torch.cuda.is_available():
        result = "cuda"
    elif torch.backends.mps.is_available():
        result = "mps"
    else:
        result = "cpu"
    return result


@dataclasses.dataclass
class _GenerationInputs:
    """Dataclass to encapsulate inputs for calling the Transformers generate() method"""

    generation_config: "transformers.GenerationConfig"

    # Re-enable to use constrained decoding
    # prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]
    model_input: dict


@backend(
    "transformers",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
            "device": {"type": "string"},
        }
    },
)
class TransformersBackend(Backend):
    _model_str: str
    _torch_device_name: str
    "Device used for hardware acceleration by this backend."
    _model: "transformers.AutoModelForCausalLM"
    _tokenizer: "transformers.AutoTokenizer"
    _executor: concurrent.futures.ThreadPoolExecutor
    "Single background thread, wrapped in an executor for queueing"

    def __init__(self, config: aconfig.Config | None = None, **kwargs):
        """
        :param model_to_load: String that the Transformers library can map to a location
            from which to load model files. This is usually a set of HuggingFace model
            coordinates such as "ibm-granite/granite-3.0-8b-instruct", but it can also
            be a local filesystem path or a URL.
        :param device: Optional Pytorch device string, usually "cpu" or "cuda"
        """
        super().__init__(config, **kwargs)

        with import_optional("transformers"):
            # Third Party
            import torch  # noqa: F401
            import transformers

        self._torch_device_name = config.device if config.device else _detect_hw_accel()
        if self._torch_device_name == "cpu":
            # CPU mode; prevent thrashing
            torch.set_num_threads(4)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name
        ).to(self._torch_device_name)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
        self._executor = concurrent.futures.ThreadPoolExecutor()

    async def pipeline(self, inputs: GenerateInputs) -> GenerateResults:
        """
        Process input, call completion (generate), process and return output

        Override to pass inputs for prefix tokens trimming.
        """
        inputs = self.process_input(inputs)
        outputs = await self.generate(inputs)
        return self.process_output(outputs=outputs, inputs=inputs)

    async def generate(self, inputs: _GenerationInputs) -> GenerateResults:
        # Wrap the call to the non-async Transformers library in a thread pool
        concurrent_futures_future = self._executor.submit(
            self._generate_callback,
            inputs,
        )
        async_future = asyncio.wrap_future(concurrent_futures_future)
        return await async_future

    def process_input(self, inputs: GenerateInputs) -> GenerateInputs:
        """Subroutine that encapsulates all the prerequisites
        that are necessary to call ``AutoModelForCausalLM.generate()``."""

        kwargs = super().process_input(inputs).model_dump(exclude_unset=True)

        # Migrate alias kwargs to this flavor of backend
        self.kwarg_alias(kwargs, "stop_strings", "stop")
        self.kwarg_alias(kwargs, "num_return_sequences", "n")

        # num_beams > num_return_sequences
        n = kwargs.get("num_return_sequences", 1)
        if n is not None and n < 1:
            raise ValueError(f"Invalid value for num_return_sequences ({n})")
        if n is not None:
            num_beams = kwargs.get("num_beams", 1)
            if num_beams is None or num_beams < n:
                num_beams = n

        # Import packages from extras "transformers"
        with import_optional("transformers"):
            # Third Party
            import torch  # noqa: F401
            import transformers

        # Turn the conversation prefix into tokens for input to the model.
        model_input = self._tokenizer(
            # The conversation up to this point
            kwargs.get("prompt"),
            # Tell the tokenizer to return a tensor instead of a Python list.
            # The model expects to receive tensors as input, so you almost always
            # need to set this.
            # You must manually select a framework. Good luck switching frameworks
            # afterwards. Here we use PyTorch.
            # Enabling tensor output also CHANGES THE SHAPE OF THE OUTPUT from
            # a 1D list to a 2D singleton batch. The model can only consume batches.
            # The tokenizer's decode() method can only consume 1D lists.
            # This tensor will be created on the framework's current default
            # device, so be sure to set that default appropriately.
            return_tensors="pt",
        )

        # AutoTokenizer uses two different tokenizer classes internally. One
        # of these classes has the ability to put tensors on the device they
        # should be on, while the other does not. Since we can't predict which
        # implementation we'll have, we need to assume that everything's on
        # the wrong device and move it to the right device. This of course
        # requires transforming the values under the keys of a dictionary.
        model_input = {
            k: v.to(self._torch_device_name) if isinstance(v, torch.Tensor) else v
            for k, v in model_input.items()
        }

        # transformers needs the tokenizer when stop_strings are given
        if kwargs.get("stop_strings"):
            model_input["tokenizer"] = self._tokenizer

        # The generate() method sometimes needs to know what is the integer ID
        # of the padding token, and for some reason this critical piece of information
        # isn't included in the serialized model. We get it from the tokenizer.
        # And of course some tokenizers don't set this parameter, in which case
        # we use the end of string token and hope for the best.
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        if pad_token_id is None:
            # Raise an error here because the some branches of the generate
            # method won't complain about an invalid value of this parameter,
            # while others will raise a cryptic exception from deep within
            # their beam search code.
            raise ValueError(
                f"Couldn't figure out padding token for tokenizer {self._tokenizer}"
            )

        # The supported way to pass parameters to the generate() method is
        # to pass them to the constructor for another class, then pass the
        # resulting object to model.generate().
        generation_config = transformers.GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", 1024),  # TEMPORARY
            num_return_sequences=n,
            num_beams=num_beams,
            # Scores are wrong anyhow, so don't bother
            output_scores=False,
            # If you don't set this flag, you'll get a string back instead of
            # a collection of multiple tensors and lists.
            return_dict_in_generate=True,
            # VERY important parameter, with no documentation on what's the right value.
            # The right value varies by model and by application.
            # Wrong values (often including the default) will produce very bad output.
            # LOWER values result in MORE penalty for repetition, because of course they
            # do.
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 50),
            temperature=kwargs.get("temperature", 1.0),
            # See long note above.
            pad_token_id=pad_token_id,
            # Make sure you specify this token explicitly, or you will have
            # a bad time.
            eos_token_id=self._tokenizer.eos_token_id,
            stop_strings=kwargs.get("stop_strings"),
        )

        # Parameters for constrained generation are **not** passed to generate()
        # via the GenerationConfig object, but are instead passed in as a
        # separate argument to the generate() method, containing a callback
        # that itself must contain a pointer to the model's tokenizer.

        # Re-enable to use constrained decoding
        # prefix_allowed_tokens_fn=_response_format_to_transformers_callback(
        #     tokenizer, response_format
        # )

        return _GenerationInputs(
            generation_config=generation_config,
            model_input=model_input,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )

    def process_output(self, outputs, *, inputs=None):
        # The result of generate() is of course the prompt concatenated with the
        # additional tokens generated. Strip off the prompt.
        generated_results = []

        # Generation inputs are needed for special processing below
        input_len = len(inputs.model_input["input_ids"][0]) if inputs else 0

        # for i, sequence in enumerate(model_output.sequences):
        for sequence in outputs.sequences:
            full_token_sequence = sequence.cpu().tolist()
            generated_tokens = full_token_sequence[input_len:]

            # The generate() method doesn't explicitly tell us why it stopped
            # generating. We are supposed to infer that from the output.
            if generated_tokens[-1] == self._tokenizer.eos_token_id:
                stop_reason = "end_of_turn"
                # We're also supposed to strip off the end-of-turn tokens ourselves.
                generated_tokens = generated_tokens[:-1]

                # When one requests multiple completions, the shorter completions will
                # come out padded with extra end-of-turn tokens so that everything is
                # the same length.
                while (
                    len(generated_tokens) > 0
                    and generated_tokens[-1] == self._tokenizer.eos_token_id
                ):
                    generated_tokens = generated_tokens[:-1]

            else:
                stop_reason = "out_of_tokens"

            # Of course, the model does not have a pointer to its tokenizer, so
            # we need to post-process the model's output to get a usable string.
            completion_string = self._tokenizer.decode(generated_tokens)

            # Now we can see if the stop reason was actually a stop string.
            if stop_strings := inputs.generation_config.stop_strings:
                if isinstance(stop_strings, str):
                    stop_strings = [stop_strings]
                for s in stop_strings:
                    if completion_string.endswith(s):
                        completion_string = completion_string.removesuffix(s)
                        stop_reason = "stop"
                        break

            generated_results.append(
                GenerateResult(
                    completion_string=completion_string,
                    completion_tokens=generated_tokens,
                    stop_reason=stop_reason,
                )
            )

        return GenerateResults(results=generated_results)

    def _generate_callback(self, generation_inputs: _GenerationInputs) -> Any:
        """
        :param generation_inputs: Wrapper for the required inputs to a Transformers
            :func:`generate()` call, prepared in the event loop thread.

        :returns: Whatever type this model's `generate()` method happens to return on
            this day of the week, which changes without warning from one version to the
            next of the library.
        """
        with import_optional("transformers"):
            # Third Party
            import torch

        def do_generate():
            return self._model.generate(
                **(generation_inputs.model_input),
                generation_config=generation_inputs.generation_config,
            )

        if self._torch_device_name == "cuda":
            # Make sure computations for this thread will happen in a separate CUDA
            # context.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                return do_generate()
        else:
            return do_generate()
