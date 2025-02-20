from typing import List, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer

from .base_aligner import BaseModelAligner
from .embeddings_cache import EmbeddingCache
from ..main import SAVE_EVERY


class EmbeddingAligner(BaseModelAligner):
    def __init__(
            self,
            model_name: str,
            threshold: float = 0.7,
            device: str = "cpu",
            cache_path: str = None
    ):
        """
        :param model_name: The sentence-transformers model name or path.
        :param threshold: Cosine similarity threshold for alignment.
        :param device: 'cpu' or 'cuda'.
        :param cache_path: If given, path to load/save the embedding cache.
        """
        super().__init__(threshold, device, cache_path)
        self.model = SentenceTransformer(model_name, device=device)

        self.cache = EmbeddingCache(cache_path)
        self._processed_count = 0

    def _encode_items(
        self,
        system_claims: List[str],
        reference_acus: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Produce or retrieve embeddings for each claim and reference.
        """

        # Batch encode both sets
        sys_embeddings = self._batch_get_embeddings(system_claims)
        ref_embeddings = self._batch_get_embeddings(reference_acus)
        return sys_embeddings, ref_embeddings

    def _compute_score(
        self,
        sys_rep: np.ndarray,
        ref_rep: np.ndarray
    ) -> float:
        """Compute cosine similarity in [0..1] range."""
        return float(
            np.dot(sys_rep, ref_rep)
            / (np.linalg.norm(sys_rep) * np.linalg.norm(ref_rep) + 1e-9)
        )

    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Retrieves or computes embeddings for a batch of texts while leveraging caching.
        """
        embeddings = []
        to_encode = []
        to_encode_idxs = []

        # 1) Look up each text in cache
        for idx, txt in enumerate(texts):
            cached = self.cache.get_embedding(txt)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_encode.append(txt)
                to_encode_idxs.append(idx)

        # 2) Encode in batch if needed
        if to_encode:
            new_embs = self.model.encode(to_encode, device=self.device, show_progress_bar=True)
            # 3) Store them in cache
            for i, emb in enumerate(new_embs):
                actual_idx = to_encode_idxs[i]
                txt = to_encode[i]
                self.cache.set_embedding(txt, emb)
                embeddings[actual_idx] = emb

        self._processed_count += len(texts)
        if self._processed_count % SAVE_EVERY == 0:
            self.cache.save_cache()

        return embeddings

    def save_alignment_cache(self):
        """Persist embedding cache to disk if needed."""
        self.cache.save_cache()
