from enum import StrEnum
from dataclasses import dataclass


@dataclass
class RosePaths:
    dataset_path: str = "src/rose/rose_datasets.json.gz"
    output_path: str = "src/rose/enhanced_rose_datasets.json.gz"


@dataclass(frozen=True)
class DatasetName:
    CNNDM_TEST: str = "cnndm_test"
    CNNDM_VALIDATION: str = "cnndm_validation"
    XSUM: str = "xsum"
    SAMSUM: str = "samsum"


MODELS = {
    "distilled_t5": {
        "name": "Babelscape/t5-base-summarization-claim-extractor",
        "claims_field": "system_claims_t5",
    },
    # Add more models as needed
    # "another_model": {
    #     "name": "some/other-model",
    #     "claims_field": "system_claims_other",
    # },
}

