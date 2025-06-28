# SPDX-License-Identifier: Apache-2.0

"""Various utility functions relating to the Granite 3.3 RAG Agent Library."""

# Standard
from collections.abc import Iterable
import dataclasses
import enum
import pathlib


@dataclasses.dataclass
class RagAgentModelInfoMixin:
    short_name: str
    """Short name of model, also used for source code and model file names."""

    long_name: str
    """Long, human-readable name of the model"""

    is_lora: bool
    """``True`` if the model is implemented as a LoRA adapter. Each LoRA adapter is 
    packaged in a subdirectory with the name ``<short_name>_lora``"""


class RagAgentLibModelInfo(RagAgentModelInfoMixin, enum.Enum):
    """Names of models within the [Granite 3.3 RAG Agent Library](
        https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib).
    Individual models are implemented as LoRA adapters on top of Granite 3.3 8B.
    """

    ANSWERABILITY_PREDICTION = (
        "answerability_prediction",
        "LoRA Adapter for Answerability Classification",
        True,
    )
    CERTAINTY = "certainty", "Granite 3.3 8B Instruct - Uncertainty LoRA", True
    HALLUCINATION = (
        "hallucination_detection",
        "LoRA Adapter for Hallucination Detection in RAG outputs",
        True,
    )
    CITATION_GENERATION = (
        "citation_generation",
        "LoRA Adapter for Citation Generation",
        True,
    )
    QUERY_REWRITE = (
        "query_rewrite",
        "LoRA Adapter for Query Rewrite",
        True,
    )
    CONTEXT_RELEVANCY = (
        "context_relevancy",
        "LoRA Adapter for Context Relevancy",
        True,
    )

    @staticmethod
    def from_str(short_name: str):
        """Retrieve metadata about a model from the model's short name."""
        for model_info in RagAgentLibModelInfo:
            if short_name == model_info.short_name:
                return model_info
        raise ValueError(
            f"No model info found for model name '{short_name}'. "
            f"Available names: "
            f"{[m.short_name for m in RagAgentLibModelInfo]}"
        )


RAG_AGENT_LIB_REPO_ID = "ibm-granite/granite-3.3-8b-rag-agent-lib"


def obtain_lora(model_name: str, cache_dir: str | None = None) -> pathlib.Path:
    """
    Downloads a cached copy of a LoRA adapter from the [Granite 3.3 RAG Agent Library](
    https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib) if one is not
    already in the local cache.  Returns the full path to the local copy of
    a specific LoRA adapter. This path is suitable for passing to commands that will
    serve the LoRA adapter.

    :param model_name: Short model name, such as "certainty". See
        :class:`RagAgentLibModelInfo` for a list of names.
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.
    """
    # Third Party
    import huggingface_hub

    model_info = RagAgentLibModelInfo.from_str(model_name)
    lora_subdir_name = f"{model_info.short_name}_lora"

    # Download just the files for this LoRA if not already present
    local_root_path = huggingface_hub.snapshot_download(
        repo_id=RAG_AGENT_LIB_REPO_ID,
        allow_patterns=f"{lora_subdir_name}/*",
        cache_dir=cache_dir,
    )

    return pathlib.Path(local_root_path) / lora_subdir_name


def obtain_loras(
    model_names: Iterable[str], cache_dir: str | None = None
) -> list[tuple[str, pathlib.Path]]:
    """
    Convenience function that calls :func:`obtain_lora()` multiple times to download
    multiple models.

    :param model_name: List/tuple of short model names, such as "certainty"
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.

    :returns: List of tuples of model name and local directory
    """
    # The Hugging Face APIs are not reentrant at all, so load models one at a time.
    return [(name, obtain_lora(name, cache_dir)) for name in model_names]
