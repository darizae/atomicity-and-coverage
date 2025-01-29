import os
import json
import gzip
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset

from src.metrics.datasets_config import DATASET_ALIASES
from src.utils.paths import RosePathsSmall, RosePaths


class RoseDatasetLoader:
    """
    A loader for the RoSE dataset subsets, with the following structure
    per record:

        {
          "source": <source text>,
          "reference": <reference (summary) text>,
          "reference_acus": {
            "original": [...],  # The original ACUs from the dataset
            # For each threshold+strategy combination, we store:
            # "deduped_0.7_longest": [<deduplicated claims>],
            # "deduped_0.7_shortest": [<deduplicated claims>],
            # etc.
          },
          "system_claims": {
            "gpt-3.5-turbo": [...],
            "distilled_t5": [...],
            ...
          }
        }

    Each dataset (e.g., cnndm_test, xsum) is stored in self.datasets[subset_name]
    as a list of such records.
    """

    def __init__(self):
        self.datasets = {}

    def load_all_datasets(self, max_entries: Optional[int] = None):
        """
        Loads all configured datasets into memory, optionally limiting the number of entries.

        :param max_entries: If set, only load up to this many entries per subset (for testing).
        :return: A dict like { "cnndm_test": [ { ... }, ... ], "xsum": [...], ... }
        """
        for alias, dataset_name in DATASET_ALIASES.items():
            print(f"Loading dataset: {alias}...")
            dataset = load_dataset(
                "Salesforce/rose",
                dataset_name,
                trust_remote_code=True
            )["data"]

            # Optionally slice to max_entries
            if max_entries is not None:
                dataset = dataset.select(range(min(max_entries, len(dataset))))

            # Structure each record
            structured_data = []
            for entry in dataset:
                record = {
                    "source": entry["source"],
                    "reference": entry["reference"],
                    "reference_acus": {
                        "original": entry["reference_acus"]
                    }
                }
                structured_data.append(record)

            self.datasets[alias] = structured_data

        return self.datasets

    def get_dataset(self, name: str):
        """
        Fetch a specific subset by name (e.g. 'cnndm_test').

        :param name: The name of the subset to fetch.
        :return: A list of records for that subset.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not loaded. Call load_all_datasets() first.")
        return self.datasets[name]

    def save_datasets_compressed(self, filepath: Path):
        """
        Saves all loaded datasets to a gzip-compressed JSON file.

        :param filepath: Path to .gz file.
        """
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(self.datasets, f)
        print(f"Datasets saved (compressed) to {filepath}.")

    def load_datasets_compressed(self, filepath: Path):
        """
        Loads datasets from a gzip-compressed JSON file into self.datasets.

        :param filepath: Path to .gz file.
        :return: The loaded datasets dict.
        """
        print(f"Current working directory: {os.getcwd()}")
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            self.datasets = json.load(f)
        print(f"Datasets loaded from {filepath}.")
        return self.datasets

    def save_datasets_json(self, filepath: Path):
        """
        Saves all datasets to a regular (non-compressed) JSON file.

        :param filepath: Path to .json file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.datasets, f, ensure_ascii=False, indent=2)
        print(f"Datasets saved (uncompressed) to {filepath}.")

    def load_datasets_json(self, filepath: Path):
        """
        Loads datasets from a .json file into self.datasets.

        :param filepath: Path to .json file.
        :return: The loaded datasets dict.
        """
        print(f"Current working directory: {os.getcwd()}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.datasets = json.load(f)
        print(f"Datasets loaded from {filepath}.")
        return self.datasets

    def add_system_claims(self, dataset_name: str, model_name: str, claims: List[List[str]]):
        """
        Add system-generated claims to a specific dataset, grouped under 'system_claims[model_name]'.

        :param dataset_name: e.g. 'cnndm_test'
        :param model_name: e.g. 'gpt-3.5-turbo', 'distilled_t5', etc.
        :param claims: A list of claim-lists, one per record.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Load it first.")

        dataset = self.datasets[dataset_name]
        if len(dataset) != len(claims):
            raise ValueError(
                f"Mismatch: dataset '{dataset_name}' has {len(dataset)} records but "
                f"provided claims has {len(claims)} sets."
            )

        for i, entry in enumerate(dataset):
            if "system_claims" not in entry:
                entry["system_claims"] = {}
            entry["system_claims"][model_name] = claims[i]

    def add_deduped_reference_acus(
        self,
        dataset_name: str,
        threshold: float,
        strategy: str,
        deduped_claims_per_record: List[List[str]]
    ):
        """
        Store deduplicated reference ACUs for each record in a dataset,
        under a key like 'deduped_{threshold}_{strategy}'.

        :param dataset_name: e.g. 'cnndm_test'
        :param threshold:  e.g. 0.7
        :param strategy:   e.g. 'longest'
        :param deduped_claims_per_record: A list of lists of strings,
                                          parallel to the dataset records.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Load it first.")

        dataset = self.datasets[dataset_name]
        if len(dataset) != len(deduped_claims_per_record):
            raise ValueError(
                f"Mismatch: dataset '{dataset_name}' has {len(dataset)} records but "
                f"provided deduped_claims has {len(deduped_claims_per_record)} lists."
            )

        threshold_str = f"{threshold:.2f}".rstrip("0").rstrip(".")
        dedup_key = f"deduped_{threshold_str}_{strategy}"

        for i, entry in enumerate(dataset):
            ref_acus = entry.setdefault("reference_acus", {})
            ref_acus[dedup_key] = deduped_claims_per_record[i]


if __name__ == "__main__":
    loader = RoseDatasetLoader()

    all_datasets = loader.load_all_datasets()
    loader.save_datasets_json(RosePaths.dataset_path)

    all_datasets_small = loader.load_all_datasets(max_entries=3)
    loader.save_datasets_json(RosePathsSmall.dataset_path)

    print("Done generating small version of the RoSE dataset.")
