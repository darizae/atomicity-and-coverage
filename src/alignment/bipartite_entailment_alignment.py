from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.optimize import linear_sum_assignment

from .base_aligner import BaseModelAligner
from .entailment_cache import NLIPredictionCache
from ..main import SAVE_EVERY


class BipartiteEntailmentAligner(BaseModelAligner):
    """
    An aligner that uses NLI-based scores to build a bipartite matching
    between system claims and reference ACUs, ensuring a one-to-one alignment.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.0,
        device: str = "cpu",
        cache_path: str = None
    ):
        """
        :param model_name: e.g. 'roberta-large-mnli'
        :param threshold: Optional minimum probability to consider an edge.
                          Scores below threshold can be masked out.
        :param device: 'cpu' or 'cuda'
        :param cache_path: path to store cache if desired
        """
        super().__init__(threshold, device, cache_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.cache = NLIPredictionCache(cache_path, save_every=SAVE_EVERY)
        self._processed_count = 0

    def align(
        self,
        system_claims: List[str],
        reference_acus: List[str],
        **kwargs
    ) -> Dict[int, List[int]]:
        """
        1) Build an N x M matrix of entailment scores (N = #system_claims, M = #reference_acus).
        2) Convert to a cost matrix = 1 - score (or some transformation).
        3) Run linear_sum_assignment to get the one-to-one matching that maximizes total entailment.
        4) Build alignment_map: each system_claim_idx is matched to at most one reference_acu_idx.
        """
        N = len(system_claims)
        M = len(reference_acus)

        # Build score matrix
        score_matrix = np.zeros((N, M), dtype=np.float32)

        for i, sys_claim in enumerate(system_claims):
            for j, ref_claim in enumerate(reference_acus):
                # Entailment score
                prob_entail = self._compute_score(sys_claim, ref_claim)

                # If you want to filter out low-prob edges:
                if prob_entail < self.threshold:
                    prob_entail = 0.0

                score_matrix[i, j] = prob_entail

        # Convert to cost. The Hungarian algorithm solves "min cost" → so we do cost = 1 - score
        cost_matrix = 1.0 - score_matrix

        # Run the linear sum assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Build alignment map. Note: linear_sum_assignment returns a match for
        # each row if possible, even if that match is not a good one. We can check
        # the actual scores to skip truly zero edges if desired.
        alignment_map = {i: [] for i in range(N)}

        for row_i, col_j in zip(row_indices, col_indices):
            matched_score = score_matrix[row_i, col_j]
            # If matched_score is 0, we might skip it, i.e. treat as "no valid match".
            if matched_score > 0:
                alignment_map[row_i] = [col_j]

        return alignment_map

    def _encode_items(self, system_claims: List[str], reference_acus: List[str]):
        # Not really used here, because we do on-the-fly inference in _compute_score
        return system_claims, reference_acus

    def _compute_score(self, sys_claim: str, ref_claim: str) -> float:
        """
        Return the entailment probability. Uses a cache for speed.
        """
        prob_entail = self.cache.get_entailment_probability(sys_claim, ref_claim)
        if prob_entail is None:
            prob_entail = self._infer_entailment(sys_claim, ref_claim)
            self.cache.set_entailment_probability(sys_claim, ref_claim, prob_entail)

        self._processed_count += 1
        if self._processed_count % SAVE_EVERY == 0:
            self.cache.save_cache()

        return prob_entail

    def _infer_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Forward pass in the NLI model to get the probability of 'entailment'.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_prob = probs[0, 2].item()  # index 2 => entailment

        return entail_prob

    def save_alignment_cache(self):
        self.cache.save_cache()
