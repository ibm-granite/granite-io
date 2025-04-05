# SPDX-License-Identifier: Apache-2.0


# Standard
from collections.abc import Callable

# Local
from granite_io.io.base import ChatCompletionResults, InputOutputProcessor
from granite_io.optional import import_optional
from granite_io.types import ChatCompletionInputs


def integer_normalizer(model_outputs: list[str]) -> list[str]:
    """
    Simple normalizer for integer answers. Parses each input string to an int and
    returns the normalized number. Non-integral values turn into the string "nan".
    """
    results = []
    for model_output in model_outputs:
        try:
            results.append(str(int(model_output)))
        except ValueError:
            results.append("nan")
    return results


class MajorityVotingProcessor(InputOutputProcessor):
    """
    Implementation of basic single-step majority sampling.
    """

    _generator: InputOutputProcessor
    _normalizer: Callable
    _samples_per_completion: int

    def __init__(
        self,
        generator: InputOutputProcessor,
        normalizer: Callable[[list[str]], list[str]],
        samples_per_completion: int = 100,
    ):
        """
        :param generator: Sub-processor over which this processor should sample
        :param normalizer: Function that takes multiple results and returns normalized
            versions of said results for grouping
        :param samples_per_completion: Number of samples to take per desired completion
        """
        self._generator = generator
        self._normalizer = normalizer
        self._samples_per_completion = samples_per_completion

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        with import_optional("pandas"):
            # Third Party
            import pandas as pd

        # Step 1: Collect samples from the "generator" sub-processor
        original_n = (
            1
            if inputs.generate_inputs is None or inputs.generate_inputs.n is None
            else inputs.generate_inputs.n
        )
        num_samples = self._samples_per_completion * original_n
        base_results = await self._generator.acreate_chat_completion(
            inputs.with_addl_generate_params({"n": num_samples})
        )

        # Step 2: Normalize the samples
        sample_contents = [s.next_message.content for s in base_results.results]
        normalized_contents = self._normalizer(sample_contents)
        assert len(sample_contents) == len(normalized_contents)  # Sanity check

        # Step 3: Group and aggregate
        indices = (
            # Group results by normalized contents, sort the groups by count, and use
            # the first result in each group as the representative of the group
            pd.DataFrame(
                {
                    "result_num": range(len(sample_contents)),
                    "normalized_result": normalized_contents,
                }
            )
            .groupby("normalized_result")
            .agg({"normalized_result": "count", "result_num": "min"})
            .rename(columns={"normalized_result": "count"})
            .sort_values(["count", "result_num"], ascending=False)
        )["result_num"].to_list()

        # If the caller requested multiple completions, return multiple results that
        # each have different normalized forms.
        num_outputs = min(len(indices), original_n)

        return ChatCompletionResults(
            results=[base_results.results[indices[i]] for i in range(num_outputs)]
        )
