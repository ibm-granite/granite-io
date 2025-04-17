# SPDX-License-Identifier: Apache-2.0


# Standard
from collections.abc import Callable
from functools import lru_cache
from typing import Union

# Third Party
from rouge_score import rouge_scorer

# Local
from granite_io.io.base import ChatCompletionResults, InputOutputProcessor
from granite_io.types import ChatCompletionInputs

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _minimum_bayesian_risk_decoding(
    answers: list[str], similarity: Callable[[str, str], float]
) -> Union[str, int, list[float]]:
    """Minimum Bayes Risk (MBR) decoding is a method for choosing the outputs of a
    machine learning system based not on the output with the highest probability, but
    the output with the lowest risk (expected error) among multiple candidates.

    In this function, it will compare each of the completion answers against each other,
    to ascertain the most similar answers.

    :param answers: List of model answers for a prompt
    :param similarity: Similarity algoritm

    :returns: Model answer most suitable
    :returns: Index from list of answers where majority answer is
    :returns: Similarity scores for all answers
    """

    similarity_scores: list[float] = []
    for _, x in enumerate(answers):
        # compute score for this answer
        score = 0
        for _, y in enumerate(answers):
            score += similarity(y, x)
        # normalization is not needed, but it gives more intepretable scores
        score = score / len(answers)
        # store
        similarity_scores.append(score)

    # select the element with highest similarity to any other
    final_index = max(range(len(answers)), key=lambda i: similarity_scores[i])
    return answers[final_index], final_index, similarity_scores


@lru_cache
def _rouge_similarity(hypotheis: str, reference: str) -> float:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used for
    evaluating the similarity between different model outputs.

    :param hypotheis: Item to compare
    :param reference: Item to compare against

    :returns: Similarity scores for the comparison
    """
    return scorer.score(hypotheis, reference)["rougeL"].fmeasure


class MBRDMajorityVotingProcessor(InputOutputProcessor):
    """
    Implementation of MBRD decoding using ROUGE score sampling for similarity
    checking.
    """

    _generator: InputOutputProcessor

    def __init__(
        self,
        generator: InputOutputProcessor,
    ):
        """
        :param generator: Sub-processor over which this processor should sample
        """
        self._generator = generator

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        # Step 1: Collect samples from the "generator" sub-processor
        base_results = await self._generator.acreate_chat_completion(inputs)

        # Step 2: Extract the content
        sample_contents = [s.next_message.content for s in base_results.results]

        # Step 3: Perform MBRD decoding and ROUGE scoring
        _, result_index, _ = _minimum_bayesian_risk_decoding(
            sample_contents, _rouge_similarity
        )

        return ChatCompletionResults(results=[base_results.results[result_index]])
