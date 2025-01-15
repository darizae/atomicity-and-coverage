from pathlib import Path
from dataclasses import dataclass

# Compute the base directory using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent  # Goes from config/ up to src/


@dataclass
class RosePaths:
    compressed_dataset_path: Path = BASE_DIR / "rose" / "rose_datasets.json.gz"
    compressed_dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets.json.gz"
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets.json"
    dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets.json"
    alignment_metrics_results: Path = BASE_DIR / "metrics" / "alignment_metrics_results.json"


@dataclass
class RosePathsSmall:
    compressed_dataset_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json.gz"
    compressed_dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json.gz"
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json"
    dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json"
    alignment_metrics_results: Path = BASE_DIR / "metrics" / "alignment_metrics_results_small.json"
