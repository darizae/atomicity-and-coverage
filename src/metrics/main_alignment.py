import argparse
from src.claims.device_selector import check_or_select_device
from src.config import AlignmentConfig
from src.metrics.alignment.alignment_factory import create_aligner
from src.metrics.alignment.utils import load_dataset, process_dataset, save_results


def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run alignment and compute metrics on a dataset.")
    parser.add_argument("--dataset_name", type=str, default="cnndm_test", help="Dataset to process.")
    parser.add_argument("--method", type=str, default=None, help="Alignment method (e.g., 'rouge', 'embedding', 'entailment').")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for the alignment method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--small_test", action="store_true", help="Run a small dataset for quick testing.")
    return parser.parse_args()


def main():
    args = get_args()

    # Initialize configuration with overrides if provided
    config = AlignmentConfig(
        method=args.method if args.method else AlignmentConfig().method,
        threshold=args.threshold if args.threshold is not None else AlignmentConfig().threshold,
        device=check_or_select_device(args.device)
    )

    # Log configuration for debugging
    print(f"Using Configuration: {config}")

    # Create the aligner based on the configuration
    aligner = create_aligner(config)

    # Load the dataset
    dataset = load_dataset(small_test=args.small_test)

    # Run alignment and compute metrics
    results = process_dataset(
        dataset.get(args.dataset_name, []),
        aligner,
    )

    # Save results
    save_results(
        results=results,
        small_test=args.small_test,
    )


if __name__ == "__main__":
    main()
