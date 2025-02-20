from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_aligner import BaseModelAligner
from .entailment_cache import NLIPredictionCache


class EntailmentAligner(BaseModelAligner):
    """
    Uses an NLI model to compute entailment probabilities. If >= threshold => match.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.9,
        device: str = "cpu",
        cache_path: str = None
    ):
        super().__init__(threshold, device, cache_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.cache = NLIPredictionCache(cache_path, save_every=500)
        self._processed_count = 0

    def _encode_items(
        self,
        system_claims: List[str],
        reference_acus: List[str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Because we do pairwise NLI, we often handle premise & hypothesis in pairs.
        Here, let's just keep track of (premise, hypothesis) pairs
        in a lazy manner. We'll do the actual 'batch_infer' in _compute_score.
        """

        # We'll store the raw text for system claims + reference claims
        # then compute probabilities on-demand in _compute_score.
        return system_claims, reference_acus

    def _compute_score(
        self,
        sys_rep: str,
        ref_rep: str
    ) -> float:
        """
        Retrieve or compute the entailment probability for (sys_rep, ref_rep).
        Then just return that probability as the alignment score.
        """
        prob_entail = self.cache.get_entailment_probability(sys_rep, ref_rep)
        if prob_entail is None:
            prob_entail = self._infer_entailment(sys_rep, ref_rep)
            self.cache.set_entailment_probability(sys_rep, ref_rep, prob_entail)

        self._processed_count += 1
        if self._processed_count % 500 == 0:
            self.cache.save_cache()

        return prob_entail

    def _infer_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Actually run the premise–hypothesis pair through the NLI model
        to get the probability of entailment.
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
            # shape: (batch_size=1, 3)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_prob = probs[0, 2].item()  # index 2 => 'entailment' for MNLI models

        return entail_prob

    def save_alignment_cache(self):
        self.cache.save_cache()
