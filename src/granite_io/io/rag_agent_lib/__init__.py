# SPDX-License-Identifier: Apache-2.0


# Local
from .util import RagAgentLibModelInfo, obtain_lora, obtain_loras
from granite_io.io.answerability.answerability import (
    AnswerabilityCompositeIOProcessor,
    AnswerabilityIOProcessor,
)
from granite_io.io.certainty.certainty import (
    AssistantMessageWithScore,
    CertaintyCompositeIOProcessor,
    CertaintyIOProcessor,
)

# This file is the root package for IO processors for the Granite 3.3 RAG Agent Library
# models.
# See https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib

# TODO: Citations
# TODO: Hallucinations
# TODO: Query rewrite


# Expose public symbols at `granite_io.io.rag_agent_lib` to save users from typing
__all__ = [
    "AssistantMessageWithScore",
    "AnswerabilityCompositeIOProcessor",
    "AnswerabilityIOProcessor",
    "CertaintyCompositeIOProcessor",
    "CertaintyIOProcessor",
    "RagAgentLibModelInfo",
    "obtain_lora",
    "obtain_loras",
]
