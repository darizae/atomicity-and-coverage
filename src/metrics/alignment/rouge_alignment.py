from typing import List, Dict
from collections import defaultdict
from rouge_score import rouge_scorer

from .base_aligner import BaseAligner


class RougeAligner(BaseAligner):
    """
    Uses the python `rouge-score` library for matching. We compare
    each system claim against each reference ACU, looking at ROUGE-1 (F-measure).
    """

    def __init__(self, threshold: float = 0.3):
        """
        :param threshold: Minimum ROUGE-1 F-measure to consider a match.
        """
        self.threshold = threshold
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def align(
            self,
            system_claims: List[str],
            reference_acus: List[str],
            **kwargs
    ) -> Dict[int, List[int]]:
        alignment_map = defaultdict(list)

        for i, s_claim in enumerate(system_claims):
            for j, r_acu in enumerate(reference_acus):
                scores = self.scorer.score(r_acu, s_claim)
                rouge1_f = scores["rouge1"].fmeasure
                if rouge1_f >= self.threshold:
                    alignment_map[i].append(j)

        return dict(alignment_map)
