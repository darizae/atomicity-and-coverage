import argparse

from src.claims.device_selector import check_or_select_device
from src.metrics.alignment.alignment_factory import create_aligner
from src.metrics.alignment.utils import load_config, load_dataset, process_dataset, save_results


def get_args():
    parser = argparse.ArgumentParser(description="Run alignment and compute metrics on a dataset.")
    parser.add_argument("--dataset_name", type=str, default="cnndm_test", help="Dataset to process.")
    parser.add_argument("--method", type=str, default="rouge", help="Alignment method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--small_test", action="store_true", help="Run a small dataset for quick testing.")
    return parser.parse_args()


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
    save_results(
        results=results,
        small_test=args.small_test
    )


if __name__ == "__main__":
    main()
