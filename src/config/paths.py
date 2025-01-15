from pathlib import Path
from dataclasses import dataclass

# Compute the base directory using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent  # Goes from config/ up to src/


@dataclass
class RosePaths:
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets.json.gz"
    output_path: Path = BASE_DIR / "rose" / "rose_datasets.json.gz"


@dataclass
class RosePathsSmall:
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json.gz"
    output_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json.gz"
