# SPDX-License-Identifier: Apache-2.0

"""
Base class for backends
"""

# Standard
import abc

# Local
from granite_io.factory import FactoryConstructible
from granite_io.types import ChatCompletionInputs, GenerateResult


# We put this base class here to avoid circular imports
class Backend(FactoryConstructible):
    """
    Base class for classes that provide an interface to a string-based completions API
    for a model.

    This base class exists for two reasons: It smoothes out the differences between
    APIs, and it provides a level of indirection so that backend inference libraries can
    be optional dependencies of this Python package.
    """

    @abc.abstractmethod
    async def generate(self, input_str: str) -> GenerateResult:
        """
        Callback to invoke the model to generate a response.
        """
        raise NotImplementedError()


class ChatCompletionBackend(FactoryConstructible):
    """
    Base class for classes that provide an interface to a chat completions API.

    This base class exists for two reasons: It smooths out the differences between
    APIs, and it provides a level of indirection so that backend inference libraries can
    be optional dependencies of this Python package.
    """

    @abc.abstractmethod
    async def create_chat_completion(self, input_chat: ChatCompletionInputs) -> str:
        """
        Callback to invoke the model to generate a response.
        """
        raise NotImplementedError()
