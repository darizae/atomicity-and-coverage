from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseAligner(ABC):
    """
    Abstract base class for all alignment strategies.
    Each aligner must implement an `align()` method
    that returns a dict of { sys_claim_idx: [matched_ref_acu_idx, ...], ... }.
    """

    @abstractmethod
    def align(
            self,
            system_claims: List[str],
            reference_acus: List[str],
            **kwargs
    ) -> Dict[int, List[int]]:
        pass


class BaseModelAligner(BaseAligner):
    """
    Intermediate class for alignment methods that rely on a model (e.g., embeddings or NLI).
    """

    def __init__(
        self,
        threshold: float,
        device: str = "cpu",
        cache_path: str = None
    ):
        """
        :param threshold: Score threshold for alignment decisions.
        :param device: Device string ('cpu' or 'cuda'), if relevant.
        :param cache_path: Optional path for caching model inferences.
        """
        self.threshold = threshold
        self.device = device
        self.cache_path = cache_path

        # Subclasses typically set these to actual instances:
        self.model = None
        self.cache = None

    def align(
        self,
        system_claims: List[str],
        reference_acus: List[str],
        **kwargs
    ) -> Dict[int, List[int]]:
        """
        Main alignment pipeline for model-based approaches:
        1. Encode system_claims, reference_acus into model representations.
        2. For each system_claim vs. each reference_acu, compute a score.
        3. If the score >= threshold, record the reference_acu index as a match.

        Returns a dict with *all* system-claim indices (including those with no matches).
        """
        # 1) Encode all items
        sys_reps, ref_reps = self._encode_items(system_claims, reference_acus)

        # 2) Pre-initialize the alignment_map so that *every* claim index appears
        alignment_map = {i: [] for i in range(len(system_claims))}

        # 3) Compute scores and build up alignment_map
        for i, sys_rep in enumerate(sys_reps):
            for j, ref_rep in enumerate(ref_reps):
                score = self._compute_score(sys_rep, ref_rep)
                if score >= self.threshold:
                    alignment_map[i].append(j)

        return alignment_map

    @abstractmethod
    def _encode_items(
        self,
        system_claims: List[str],
        reference_acus: List[str]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Convert system_claims and reference_acus into model-specific representations
        (e.g., embeddings, tokenized inputs, etc.) and return them.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_score(
        self,
        sys_rep: Any,
        ref_rep: Any
    ) -> float:
        """
        Given the model-specific representations (sys_rep, ref_rep), compute a similarity
        or probability score in [0..1] that indicates how well they match.
        """
        raise NotImplementedError

    def save_alignment_cache(self):
        """
        Optional method to save to disk. Subclasses can override if they maintain a cache.
        """
        pass
