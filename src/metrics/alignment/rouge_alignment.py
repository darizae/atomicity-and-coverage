from typing import List, Dict
from rouge_score import rouge_scorer

from .base_aligner import BaseAligner


class RougeAligner(BaseAligner):
    """
    Uses ROUGE-1 F-measure to decide if system_claim matches reference_ACU.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def align(
        self,
        system_claims: List[str],
        reference_acus: List[str],
        **kwargs
    ) -> Dict[int, List[int]]:
        # Pre-initialize alignment_map with an empty list for each claim
        alignment_map = {i: [] for i in range(len(system_claims))}

        for i, s_claim in enumerate(system_claims):
            for j, r_acu in enumerate(reference_acus):
                scores = self.scorer.score(target=r_acu, prediction=s_claim)
                rouge1_f = scores["rouge1"].fmeasure
                if rouge1_f >= self.threshold:
                    alignment_map[i].append(j)

        return alignment_map
