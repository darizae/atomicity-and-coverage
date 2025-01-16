import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.config import RosePaths, RosePathsSmall
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage


def _load_dataset(small_test: bool = False):
    # Determine the path based on dataset subset
    paths = RosePathsSmall() if small_test else RosePaths()
    dataset_path = paths.compressed_dataset_path

    loader = RoseDatasetLoader()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
    loader.load_datasets_compressed(dataset_path)

    return loader.datasets if not small_test else {k: v[:1] for k, v in loader.datasets.items()}


def _process_dataset(dataset, aligner):
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
            "reference summary": record.get("reference", ""),
            "coverage": coverage,
            "atomicity": atomicity,
            "alignment_map": alignment_map
        })
    return results


def process_all_datasets(
        datasets: List[str],
        aligner,
        small_test: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all datasets and aggregate results into a single structure.

    Args:
        datasets (List[str]): List of dataset names to process.
        aligner: The alignment model.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Combined results from all datasets.
    """
    combined_results = {}
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
        results = _process_dataset(dataset, aligner)
        combined_results[dataset_name] = results
        print(f"Finished processing dataset: {dataset_name}")
    return combined_results


def process_single_dataset(
        dataset_name: str,
        aligner,
        small_test: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a single dataset.

    Args:
        dataset_name (str): Name of the dataset to process.
        aligner: The alignment model.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        List[Dict[str, Any]]: Results for the single dataset.
    """
    print(f"Processing single dataset: {dataset_name}")
    dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
    return _process_dataset(dataset, aligner)


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


def save_all_results(all_results: Dict[str, List[Dict[str, Any]]], small_test: bool = False) -> None:
    """
    Save all dataset results to a single JSON file.

    Args:
        all_results (Dict[str, List[Dict[str, Any]]]): The combined results from all datasets.
        small_test (bool): Whether to use the small dataset path variant.
    """
    paths = RosePathsSmall() if small_test else RosePaths()
    output_path = paths.alignment_metrics_results.with_name("combined_alignment_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {output_path}")
