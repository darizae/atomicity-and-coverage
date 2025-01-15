from src.config import RosePathsSmall, AlignmentConfig, DatasetName
from src.metrics.atomicity_coverage import compute_coverage, compute_atomicity
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.alignment import match_claims


def main():
    # Load configurations
    dataset_config = RosePathsSmall()
    alignment_config = AlignmentConfig()

    # Initialize the loader and load datasets
    loader = RoseDatasetLoader()
    datasets = loader.load_datasets_compressed(dataset_config.dataset_path)

    # Choose dataset
    dataset_name = DatasetName.CNNDM_TEST
    if dataset_name not in datasets:
        print(f"Dataset '{dataset_name}' not found in loaded datasets.")
        return
    dataset = datasets[dataset_name]

    # Process each record
    results = []
    for record in dataset:
        reference_acus = record.get("reference_acus", [])
        system_claims = record.get("system_claims_t5", [])

        if not reference_acus or not system_claims:
            print("Skipping record with missing claims or references.")
            continue

        # Perform alignment
        alignment_map = match_claims(
            system_claims=system_claims,
            reference_acus=reference_acus,
            threshold=alignment_config.threshold,
            method=alignment_config.method,
            device=alignment_config.device
        )

        # Compute metrics
        coverage_score = compute_coverage(alignment_map, len(reference_acus))
        atomicity_score = compute_atomicity(alignment_map, len(system_claims))

        # Store results
        results.append({
            "coverage": coverage_score,
            "atomicity": atomicity_score,
            "alignment_map": alignment_map
        })

    # Save results
    # Get the current filename and perform string replacement
    new_filename = dataset_config.output_path.name.replace(".json.gz", "_metrics.json")
    # Create a new Path with the modified filename in the same directory
    output_path = dataset_config.output_path.with_name(new_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Metrics results saved to {output_path}")


if __name__ == "__main__":
    main()
