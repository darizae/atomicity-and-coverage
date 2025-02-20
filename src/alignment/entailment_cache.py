import pickle
import os
from typing import Tuple, Dict


class NLIPredictionCache:
    """
    A dictionary from (premise, hypothesis) -> entailment_probability
    """

    def __init__(self, cache_path: str = None, save_every: int = 100):
        self.cache_path = cache_path
        self._cache: Dict[Tuple[str, str], float] = {}
        self.save_every = save_every
        self._counter = 0
        if cache_path and os.path.exists(cache_path):
            self.load_cache(cache_path)

    def get_entailment_probability(self, premise: str, hypothesis: str):
        return self._cache.get((premise, hypothesis))

    def set_entailment_probability(self, premise: str, hypothesis: str, probability: float):
        self._cache[(premise, hypothesis)] = probability
        self._counter += 1
        if self._counter % self.save_every == 0:
            self.save_cache()

    def save_cache(self, path: str = None):
        path = path or self.cache_path
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._cache, f)
        print(f"NLI cache saved to {path}.")

    def load_cache(self, path: str = None):
        path = path or self.cache_path
        if not path or not os.path.exists(path):
            return
        with open(path, "rb") as f:
            self._cache = pickle.load(f)
        print(f"NLI cache loaded from {path}.")
