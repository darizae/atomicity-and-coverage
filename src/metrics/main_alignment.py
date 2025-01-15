import argparse
import os
import json
from src.claims.device_selector import check_or_select_device
from src.config import RosePaths, RosePathsSmall
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage
from src.metrics.alignment.alignment_factory import create_aligner


def load_config(config_path="configs/alignment_config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Run alignment and compute metrics on a dataset.")
    parser.add_argument("--dataset_name", type=str, default="cnndm_test", help="Dataset to process.")
    parser.add_argument("--method", type=str, default="rouge", help="Alignment method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--small_test", action="store_true", help="Run a small dataset for quick testing.")
    parser.add_argument("--output", type=str, default="alignment_results.json", help="Output JSON file path.")
    return parser.parse_args()


def load_dataset(small_test=False):
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


def main():
    args = get_args()
    device = check_or_select_device(args.device)
    config = load_config()

    # Load alignment method configuration
    method_config = config.get(args.method, {})
    aligner = create_aligner(
        method=args.method,
        threshold=method_config.get("threshold", 0.3),
        device=device
    )

    # Load dataset
    dataset = load_dataset(small_test=args.small_test)

    # Run alignment and compute metrics
    results = process_dataset(
        dataset.get(
            args.dataset_name,
            []
        ),
        aligner
    )

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
