import numpy as np
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment

from sentence_transformers import SentenceTransformer

from .base_aligner import BaseModelAligner
from .embeddings_cache import EmbeddingCache
from ..main import SAVE_EVERY


class BipartiteEmbeddingAligner(BaseModelAligner):
    """
    An aligner that uses embeddings-based cosine similarity
    to do bipartite (one-to-one) matching between
    system claims and reference ACUs.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.0,
        device: str = "cpu",
        cache_path: str = None
    ):
        """
        :param model_name: e.g. 'sentence-transformers/all-MiniLM-L6-v2'
        :param threshold: min cosine similarity to treat an edge as valid
        :param device: 'cpu' or 'cuda'
        :param cache_path: optional path for caching embeddings
        """
        super().__init__(threshold, device, cache_path)

        self.model = SentenceTransformer(model_name, device=device)
        self.cache = EmbeddingCache(cache_path)
        self._processed_count = 0

    def align(
        self,
        system_claims: List[str],
        reference_acus: List[str],
        **kwargs
    ) -> Dict[int, List[int]]:
        """
        1) Compute embeddings for each system claim & reference claim
        2) Build N x M similarity matrix
        3) Convert to cost = 1 - similarity
        4) Run Hungarian algorithm
        5) Return alignment map { s_idx: [r_idx] } (one-to-one)
        """
        # 1) embeddings
        sys_embeddings = self._batch_get_embeddings(system_claims)
        ref_embeddings = self._batch_get_embeddings(reference_acus)

        N = len(sys_embeddings)
        M = len(ref_embeddings)

        # 2) Build similarity matrix
        sim_matrix = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                sim = self._cosine_sim(sys_embeddings[i], ref_embeddings[j])
                # If below threshold, set to 0 so we skip it in matching
                if sim < self.threshold:
                    sim = 0.0
                sim_matrix[i, j] = sim

        # 3) Convert sim -> cost (the Hungarian method wants min cost)
        cost_matrix = 1.0 - sim_matrix

        # 4) Solve linear sum assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 5) Build alignment map
        alignment_map = {i: [] for i in range(N)}
        for row_i, col_j in zip(row_indices, col_indices):
            matched_sim = sim_matrix[row_i, col_j]
            # If it's 0, skip it
            if matched_sim > 0.0:
                alignment_map[row_i] = [col_j]

        return alignment_map

    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Use the existing EmbeddingCache to speed up repeated calls.
        """
        embeddings = []
        to_encode = []
        to_encode_idxs = []

        for idx, txt in enumerate(texts):
            cached = self.cache.get_embedding(txt)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_encode.append(txt)
                to_encode_idxs.append(idx)

        if to_encode:
            new_embs = self.model.encode(
                to_encode,
                device=self.device,
                show_progress_bar=True
            )
            for i, emb in enumerate(new_embs):
                actual_idx = to_encode_idxs[i]
                txt = to_encode[i]
                self.cache.set_embedding(txt, emb)
                embeddings[actual_idx] = emb

        self._processed_count += len(texts)
        if self._processed_count % SAVE_EVERY == 0:
            self.cache.save_cache()

        return embeddings

    def _cosine_sim(self, vecA: np.ndarray, vecB: np.ndarray) -> float:
        denom = (np.linalg.norm(vecA) * np.linalg.norm(vecB)) + 1e-9
        return float(np.dot(vecA, vecB) / denom)

    def _encode_items(self, system_claims: List[str], reference_acus: List[str]):
        """
        Overridden from BaseModelAligner but not used here,
        because we compute embeddings on the fly in align().
        """
        return system_claims, reference_acus

    def _compute_score(self, sys_rep: Any, ref_rep: Any) -> float:
        """
        For bipartite approach, we don't use this method in the loop
        because we do everything in align() at once.
        You could implement it if you want to keep the same pattern
        but it's not strictly needed.
        """
        raise NotImplementedError(
            "BipartiteEmbeddingAligner does the entire matrix at once; "
            "no need for _compute_score()."
        )

    def save_alignment_cache(self):
        self.cache.save_cache()
