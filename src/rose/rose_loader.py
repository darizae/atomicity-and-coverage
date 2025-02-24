import os
import json
import gzip
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset

from src.metrics.datasets_config import DATASET_ALIASES
from src.utils.paths import RosePathsSmall


class RoseDatasetLoader:
    """
    A loader for the RoSE dataset and its subsets with functionality
    to save and load compressed and regular JSON datasets.
    """

    def __init__(self):
        self.datasets = {}

    def load_all_datasets(self, max_entries: Optional[int] = None):
        """
        Loads all configured datasets into memory, optionally limiting the number of entries.

        Args:
            max_entries (int, optional): If provided, only load up to this many entries
                                         per dataset for testing.
        """
        for alias, dataset_name in DATASET_ALIASES.items():
            print(f"Loading dataset: {alias}...")
            dataset = load_dataset("Salesforce/rose", dataset_name, trust_remote_code=True)["data"]

            # Structure the data, optionally slicing if max_entries is specified
            if max_entries is not None:
                dataset = dataset.select(range(min(max_entries, len(dataset))))

            structured_data = [
                {
                    "source": entry["source"],
                    "reference": entry["reference"],
                    "reference_acus": entry["reference_acus"],
                }
                for entry in dataset
            ]

            self.datasets[alias] = structured_data
        return self.datasets

    def get_dataset(self, name):
        """
        Fetches a specific dataset by name.

        Args:
            name (str): The name of the dataset to fetch.

        Returns:
            list: The requested dataset.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' has not been loaded. Use 'load_all_datasets()' first.")
        return self.datasets[name]

    def save_datasets_compressed(self, filepath: Path):
        """
        Saves all datasets to a compressed file in gzip format.

        Args:
            filepath (str): The path to the compressed file.
        """
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(self.datasets, f)
        print(f"Datasets saved to {filepath} in compressed format.")

    def load_datasets_compressed(self, filepath: Path):
        """
        Loads datasets from a compressed gzip file.

        Args:
            filepath (str): The path to the compressed file.

        Returns:
            dict: The loaded datasets.
        """
        print(f"Current working directory: {os.getcwd()}")
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            self.datasets = json.load(f)
        print(f"Datasets loaded from {filepath}.")
        return self.datasets

    def save_datasets_json(self, filepath: Path):
        """
        Saves all datasets to a regular (non-compressed) JSON file.

        Args:
            filepath (str): The path to the JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.datasets, f, ensure_ascii=False, indent=2)
        print(f"Datasets saved to {filepath} in JSON format.")

    def load_datasets_json(self, filepath: Path):
        """
        Loads datasets from a regular (non-compressed) JSON file.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: The loaded datasets.
        """
        print(f"Current working directory: {os.getcwd()}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.datasets = json.load(f)
        print(f"Datasets loaded from {filepath}.")
        return self.datasets

    def add_claims(self, dataset_name: str, claims_field: str, claims: List[List[str]]):
        """
        Add system-generated claims to a specific dataset.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Load it first.")

        dataset = self.datasets[dataset_name]
        for i, entry in enumerate(dataset):
            entry[claims_field] = claims[i]


if __name__ == "__main__":
    loader = RoseDatasetLoader()

    # 1. Load the full datasets and save them
    # all_datasets = loader.load_all_datasets()
    # loader.save_datasets_compressed(RosePaths.compressed_dataset_path)
    # loader.save_datasets_json(RosePaths.dataset_path)

    # 2. Load a SMALL version of each dataset (e.g., 3 entries) and save them
    all_datasets_small = loader.load_all_datasets(max_entries=3)
    loader.save_datasets_compressed(RosePathsSmall.dataset_path)
    loader.save_datasets_json(RosePathsSmall.dataset_path)

    print("Done generating both full and small datasets.")
