from abc import ABC, abstractmethod
from typing import List, Dict


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
