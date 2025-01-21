from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetName:
    CNNDM_TEST: str = "cnndm_test"
    CNNDM_VALIDATION: str = "cnndm_validation"
    XSUM: str = "xsum"
    SAMSUM: str = "samsum"


# Utility function for dataset aliases (optional)
DATASET_ALIASES = {
    "cnndm_test": DatasetName.CNNDM_TEST,
    "cnndm_validation": DatasetName.CNNDM_VALIDATION,
    "xsum": DatasetName.XSUM,
    "samsum": DatasetName.SAMSUM,
}
