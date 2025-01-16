from typing import List, Dict
from collections import defaultdict

from .base_aligner import BaseAligner


class EntailmentAligner(BaseAligner):
    """
    Alignment based on NLI: we check if system_claim => reference_ACU
    is entailed with probability >= threshold.
    """

    def __init__(self, nli_model, tokenizer, threshold: float = 0.9, device: str = "cpu"):
        """
        :param nli_model: A HuggingFace model for NLI, e.g. bart-large-mnli or roberta-large-mnli
        :param tokenizer: Corresponding tokenizer for the NLI model.
        :param threshold: Probability of entailment needed to consider a match.
        :param device: 'cpu' or 'cuda'.
        """
        self.nli_model = nli_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = device

    def align(
            self,
            system_claims: List[str],
            reference_acus: List[str],
            **kwargs
    ) -> Dict[int, List[int]]:
        alignment_map = defaultdict(list)

        for i, s_claim in enumerate(system_claims):
            for j, r_acu in enumerate(reference_acus):
                prob_entailment = self._compute_entailment_probability(s_claim, r_acu)
                if prob_entailment >= self.threshold:
                    alignment_map[i].append(j)

        return dict(alignment_map)

    def _compute_entailment_probability(self, premise: str, hypothesis: str) -> float:
        """
        Example approach: For a HuggingFace-based NLI model, pass premise & hypothesis,
        take the 'ENTAILMENT' logit, apply softmax, and return that probability.
        """
        import torch

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits[0]  # shape: (3,) if MNLI
            probs = torch.softmax(logits, dim=-1)
            # typically, index 2 => entailment, 1 => neutral, 0 => contradiction
            entail_prob = probs[2].item()
        return entail_prob
