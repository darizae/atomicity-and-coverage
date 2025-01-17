from typing import List, Dict
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .base_aligner import BaseAligner
from .entailment_cache import NLIPredictionCache


class EntailmentAligner(BaseAligner):
    """
    Alignment based on an NLI model. For each system_claim => reference_ACU, we compute
    the entailment probability. If >= threshold, we consider it a match.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.9,
        device: str = "cpu",
        cache_path: str = None
    ):
        """
        :param model_name: e.g. "roberta-large-mnli"
        :param threshold: Probability of entailment needed to consider a match
        :param device: 'cpu' or 'cuda'
        :param cache_path: If given, path to load/save the NLI inference cache
        """
        # 1) Load the model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nli_model.to(device)

        self.threshold = threshold
        self.device = device

        self.cache = NLIPredictionCache(cache_path)

    def align(
            self,
            system_claims: List[str],
            reference_acus: List[str],
            **kwargs
    ) -> Dict[int, List[int]]:
        # 1) Gather all pairs (system_claim, reference_ACU)
        pairs_to_infer = []
        for s_claim in system_claims:
            for r_acu in reference_acus:
                # If not in cache, we need to compute
                if self.cache.get_entailment_probability(s_claim, r_acu) is None:
                    pairs_to_infer.append((s_claim, r_acu))

        # 2) Perform batch inference on all unknown pairs
        if pairs_to_infer:
            self._batch_infer_entailment(pairs_to_infer)

        # 3) Build alignment map using cached results
        alignment_map = defaultdict(list)
        for i, s_claim in enumerate(system_claims):
            for j, r_acu in enumerate(reference_acus):
                prob_entail = self.cache.get_entailment_probability(s_claim, r_acu)
                if prob_entail is not None and prob_entail >= self.threshold:
                    alignment_map[i].append(j)

        return dict(alignment_map)

    def _compute_entailment_probability(self, premise: str, hypothesis: str) -> float:

        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits[0]  # shape: (3,) for MNLI
            probs = torch.softmax(logits, dim=-1)
            # Index 2 => entailment, 1 => neutral, 0 => contradiction for many MNLI models
            entail_prob = probs[2].item()

        return entail_prob

    def save_alignment_cache(self):
        """
        Save the NLI results to disk, so subsequent runs skip repeated computations.
        """
        self.cache.save_cache()
