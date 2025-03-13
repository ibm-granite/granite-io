# SPDX-License-Identifier: Apache-2.0

"""
Base class for backends
"""

# Standard
from typing import Any, Dict
import abc

# Local
from granite_io.factory import FactoryConstructible
from granite_io.types import ChatCompletionInputs, GenerateResults


# We put this base class here to avoid circular imports
class Backend(FactoryConstructible):
    """
    Base class for classes that provide an interface to a string-based completions API
    for a model.

    This base class exists for two reasons: It smoothes out the differences between
    APIs, and it provides a level of indirection so that backend inference libraries can
    be optional dependencies of this Python package.
    """

    # TODO: decide between __call__ and pipeline. Don't expand kwargs 2x
    async def __call__(self, **kwargs: Any) -> GenerateResults:
        return await self.pipeline(**kwargs)

    async def pipeline(self, **kwargs: Any) -> GenerateResults:
        """
        Process input, call completion (generate), process and return output
        """
        inputs = self.process_input(**kwargs)
        output = await self.generate(**inputs)
        return self.process_output(output)

    @abc.abstractmethod
    def process_input(self, **kwargs) -> Dict[str, Any]:
        """
        Process kwargs to prepare them for completion.create() (or generate())

        Args:
            kwargs (dict): This is not just the kwargs for this function, but is
             intended to be the inputs which are being processed (modified)
             to be used in target backend call.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def generate(self, **kwargs) -> GenerateResults:
        """
        Callback to invoke the model to generate a response.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def process_output(self, output, **kwargs):
        """
        Process output from completion.create() (or generate())
        """
        raise NotImplementedError()

    @staticmethod
    def kwarg_alias(kw_dict, preferred_key, alias_key):
        """Migrate kwarg k,v from alias key to preferred key"""
        if alias_key in kw_dict:
            alias_value = kw_dict.get(alias_key)
            # Alias is removed
            del kw_dict[alias_key]
            # Preferred key/value gets priority if both are specified
            # TODO: optionally error on redundant kwargs(?)
            if preferred_key not in kw_dict:
                # Alias value is set to preferred key (iff not already set)
                kw_dict[preferred_key] = alias_value
        return kw_dict  # return reference for convenience


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
