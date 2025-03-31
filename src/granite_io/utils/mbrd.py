# SPDX-License-Identifier: Apache-2.0

"""
Majority voting using Minimum Bayesian Risk Decoding (MBRD)
"""

# Standard
from functools import lru_cache

# Third Party
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def minimum_bayesian_risk_decoding(answers, similarity):
    """Minimum Bayes Risk (MBR) decoding is a method for choosing the outputs of a
    machine learning system based not on the output with the highest probability, but
    the output with the lowest risk (expected error) among multiple candidates.

    In this function, it will compare each of the completion answers against each other,
    to ascertain the most similar answers.

    :param answers: List of model answers for a prompt
    :param similarity: Similarity algoritm

    :returns: Model answer most suitable
    :returns: Similarity scores for all answers
    """

    similarity_scores = []
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
    return answers[final_index], similarity_scores


@lru_cache
def rouge_similarity(hypotheis, reference):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used for
    evaluating the similarity between different model outputs.

    :param hypotheis: Item to compare
    :param reference: Item to compare against

    :returns: Similarity scores for the comparison
    """
    return scorer.score(hypotheis, reference)["rougeL"].fmeasure
