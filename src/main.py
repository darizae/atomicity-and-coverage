import os

from src.metrics.atomicity_coverage import compute_coverage, compute_atomicity
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.alignment import match_claims


def main():
    """
    Main function to load datasets, align claims, and compute metrics.
    """

    # Step 1: Initialize the loader and load a small dataset for testing
    loader = RoseDatasetLoader()
    dataset_path = "rose/rose_datasets_small.json.gz"
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} not found. Generating small datasets...")
        loader.load_all_datasets(max_entries=3)
        loader.save_datasets_compressed(dataset_path)

    # Load the dataset
    datasets = loader.load_datasets_compressed(dataset_path)

    # Choose a dataset (e.g., "cnndm_test")
    dataset_name = "cnndm_test"
    if dataset_name not in datasets:
        print(f"Dataset '{dataset_name}' not found in loaded datasets.")
        return
    dataset = datasets[dataset_name]

    # Step 2: Iterate through the dataset and align claims
    results = []
    for record in dataset:
        source_text = record.get("source", "")
        reference_acus = record.get("reference_acus", [])
        system_claims = record.get("system_claims_t5", [])

        if not reference_acus or not system_claims:
            print("Skipping record with missing claims or references.")
            continue

        # Perform alignment
        alignment_map = match_claims(
            system_claims=system_claims,
            reference_acus=reference_acus,
            threshold=0.3,  # example threshold
            method="rouge"  # choose "rouge", "embedding", or "entailment"
        )

        # Compute metrics
        coverage_score = compute_coverage(alignment_map, len(reference_acus))
        atomicity_score = compute_atomicity(alignment_map, len(system_claims))

        # Log results
        results.append({
            "source": source_text[:80] + "...",  # Truncate for display
            "coverage": coverage_score,
            "atomicity": atomicity_score,
            "alignment_map": alignment_map
        })

        print(f"Processed record. Coverage: {coverage_score}, Atomicity: {atomicity_score}")

    # Step 3: Save or display results
    results_path = "metrics_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Metrics results saved to {results_path}")


if __name__ == "__main__":
    main()
