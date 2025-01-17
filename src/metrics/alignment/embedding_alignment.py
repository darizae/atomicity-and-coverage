from typing import List, Dict
from collections import defaultdict
import numpy as np

from .base_aligner import BaseAligner
from .embeddings_cache import EmbeddingCache

from sentence_transformers import SentenceTransformer


class EmbeddingAligner(BaseAligner):
    def __init__(
        self,
        model_name: str,           # e.g. "sentence-transformers/all-MiniLM-L6-v2"
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
        # 1) Actually load the model (rather than storing just a string):
        self.model = SentenceTransformer(model_name, device=device)

        # 2) Store the alignment threshold, device, etc.
        self.threshold = threshold
        self.device = device

        # 3) Initialize cache
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
        """
        Retrieves or computes embeddings for a batch of texts while leveraging caching.

        Workflow:
        1. Retrieve cached embeddings and collect texts that need encoding.
        2. Encode the uncached texts in batch.
        3. Store new embeddings in cache and assign them back to the final list.
        """

        # Step 1: Retrieve cached embeddings & collect texts to encode
        embeddings, texts_to_encode, idx_to_encode = self._retrieve_cached_embeddings(texts)

        # Step 2: Perform batch encoding if necessary
        batch_embs = self._encode_texts_in_batch(texts_to_encode)

        # Step 3: Update cache and assign embeddings
        self._update_cache_and_assign_embeddings(embeddings, texts_to_encode, idx_to_encode, batch_embs)

        return embeddings

    def _retrieve_cached_embeddings(self, texts: List[str]):
        """
        Retrieves cached embeddings and prepares a list of texts that need encoding.

        Returns:
        - embeddings: List with cached embeddings (or None if not cached)
        - texts_to_encode: List of texts that need encoding
        - idx_to_encode: Indices corresponding to the texts_to_encode
        """

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

        return embeddings, texts_to_encode, idx_to_encode

    def _encode_texts_in_batch(self, texts_to_encode: List[str]) -> List[np.ndarray]:
        """
        Retrieves cached embeddings and prepares a list of texts that need encoding.

        Returns:
        - embeddings: List with cached embeddings (or None if not cached)
        - texts_to_encode: List of texts that need encoding
        - idx_to_encode: Indices corresponding to the texts_to_encode
        """

        return self.model.encode(
            texts_to_encode,
            device=self.device,
            show_progress_bar=True
        ) if texts_to_encode else []

    def _update_cache_and_assign_embeddings(self, embeddings: List[np.ndarray], texts_to_encode: List[str],
                                            idx_to_encode: List[int], batch_embs: List[np.ndarray]):
        """Updates the cache with new embeddings and assigns them to the original list."""
        for i, emb in enumerate(batch_embs):
            txt = texts_to_encode[i]
            self.cache.set_embedding(txt, emb)
            embeddings[idx_to_encode[i]] = emb

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))

    def save_alignment_cache(self):
        """
        Explicit method to allow saving the alignment cache on demand.
        """
        self.cache.save_cache()
