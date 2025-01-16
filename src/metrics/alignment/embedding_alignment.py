from typing import List, Dict
from collections import defaultdict
import numpy as np

from .base_aligner import BaseAligner
from .embeddings_cache import EmbeddingCache


class EmbeddingAligner(BaseAligner):
    """
    Alignment via embedding-based similarity, e.g. sentence-transformers.
    """

    def __init__(self, model, threshold: float = 0.7, device: str = "cpu", cache_path: str = None):
        """
        :param model: A model with a `.encode()` method that returns embeddings.
        :param threshold: Minimum cosine similarity for a match.
        :param device: 'cpu' or 'cuda' for inference.
        :param cache_path: Path to load/save embeddings cache.
        """
        self.model = model
        self.threshold = threshold
        self.device = device
        self.cache = EmbeddingCache(cache_path)

    def align(
            self,
            system_claims: List[str],
            reference_acus: List[str],
            **kwargs
    ) -> Dict[int, List[int]]:
        alignment_map = defaultdict(list)

        # Batch encode both sets
        sys_embeddings = self._batch_get_embeddings(system_claims)
        ref_embeddings = self._batch_get_embeddings(reference_acus)

        # Compare each system claim embedding to each reference ACU embedding
        for i, s_emb in enumerate(sys_embeddings):
            for j, r_emb in enumerate(ref_embeddings):
                sim = self._cosine_similarity(s_emb, r_emb)
                if sim >= self.threshold:
                    alignment_map[i].append(j)

        return dict(alignment_map)

    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        texts_to_encode = []
        idx_to_encode = []

        for idx, txt in enumerate(texts):
            # Check the cache
            cached_emb = self.cache.get_embedding(txt)
            if cached_emb is not None:
                embeddings.append(cached_emb)
            else:
                embeddings.append(None)
                texts_to_encode.append(txt)
                idx_to_encode.append(idx)

        # Perform batch encoding for all unknown texts
        if texts_to_encode:
            batch_embs = self.model.encode(
                texts_to_encode,
                device=self.device,
                show_progress_bar=True
            )
            for i, emb in enumerate(batch_embs):
                txt = texts_to_encode[i]
                self.cache.set_embedding(txt, emb)
                embeddings[idx_to_encode[i]] = emb

        return embeddings

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))

    def save_alignment_cache(self):
        """
        Explicit method to allow saving the alignment cache on demand.
        """
        self.cache.save_cache()
