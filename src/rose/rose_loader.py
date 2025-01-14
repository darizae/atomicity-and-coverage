import os
import json
import gzip
from typing import List, Optional
from datasets import load_dataset


class RoseDatasetLoader:
    """
    A loader for the RoSE dataset and its subsets with functionality
    to save and load compressed and regular JSON datasets.
    """

    DATASETS_CONFIG = [
        {"name": "cnndm_test", "hf_name": "cnndm_test"},
        {"name": "cnndm_validation", "hf_name": "cnndm_validation"},
        {"name": "xsum", "hf_name": "xsum"},
        {"name": "samsum", "hf_name": "samsum"},
    ]

    def __init__(self):
        self.datasets = {}

    def load_all_datasets(self, max_entries: Optional[int] = None):
        """
        Loads all configured datasets into memory, optionally limiting the number of entries.

        Args:
            max_entries (int, optional): If provided, only load up to this many entries
                                         per dataset for testing.
        """
        for config in self.DATASETS_CONFIG:
            dataset_name = config["name"]
            hf_name = config["hf_name"]

            print(f"Loading dataset: {dataset_name}...")
            dataset = load_dataset("Salesforce/rose", hf_name)["data"]

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

            self.datasets[dataset_name] = structured_data
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

    def save_datasets_compressed(self, filepath):
        """
        Saves all datasets to a compressed file in gzip format.

        Args:
            filepath (str): The path to the compressed file.
        """
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(self.datasets, f)
        print(f"Datasets saved to {filepath} in compressed format.")

    def load_datasets_compressed(self, filepath):
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

    def save_datasets_json(self, filepath):
        """
        Saves all datasets to a regular (non-compressed) JSON file.

        Args:
            filepath (str): The path to the JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.datasets, f, ensure_ascii=False, indent=2)
        print(f"Datasets saved to {filepath} in JSON format.")

    def load_datasets_json(self, filepath):
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

    # 1. Load the full datasets
    all_datasets = loader.load_all_datasets()
    loader.save_datasets_compressed("rose_datasets.json.gz")
    loader.save_datasets_json("rose_datasets.json")

    # 2. Load a SMALL version of each dataset (e.g., 1 entry)
    all_datasets_small = loader.load_all_datasets(max_entries=3)
    loader.save_datasets_compressed("rose_datasets_small.json.gz")
    loader.save_datasets_json("rose_datasets_small.json")

    print("Done generating both full and small datasets.")
