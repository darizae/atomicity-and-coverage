import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.config import RosePaths, RosePathsSmall
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage


def load_dataset(small_test: bool = False):
    # Determine the path based on dataset subset
    paths = RosePathsSmall() if small_test else RosePaths()
    dataset_path = paths.compressed_dataset_path

    loader = RoseDatasetLoader()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
    loader.load_datasets_compressed(dataset_path)

    return loader.datasets if not small_test else {k: v[:1] for k, v in loader.datasets.items()}


def process_dataset(dataset, aligner):
    results = []
    for record in dataset:
        system_claims = record.get("system_claims_t5", [])
        reference_acus = record.get("reference_acus", [])
        if not system_claims or not reference_acus:
            continue

        alignment_map = aligner.align(system_claims, reference_acus)
        coverage = compute_coverage(alignment_map, len(reference_acus))
        atomicity = compute_atomicity(alignment_map, len(system_claims))

        results.append({
            "source": record.get("source", "")[:80] + "...",
            "coverage": coverage,
            "atomicity": atomicity,
            "alignment_map": alignment_map
        })
    return results


def save_results(results: List[Dict[str, Any]], small_test: bool = False) -> None:
    """
    Save the results to a JSON file.

    Args:
        results (List[Dict[str, Any]]): The results to save.
        small_test (str): Path to the output JSON file.
    """

    paths = RosePathsSmall() if small_test else RosePaths()
    output_path = paths.alignment_metrics_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
