import argparse
from src.claims.device_selector import check_or_select_device
from src.config import AlignmentConfig, DatasetName
from src.config.alignment_config import EmbeddingModelConfig
from src.config.models_config import EMBEDDING_MODELS
from src.metrics.alignment.alignment_factory import create_aligner
from src.metrics.alignment.utils import save_results, process_single_dataset, \
    process_all_datasets, save_all_results
from src.utils.timer import Timer


def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run alignment and compute metrics on a dataset.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,  # No default dataset specified
        help="Dataset to process. If not provided, all datasets will be processed."
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Alignment method (e.g., 'rouge', 'embedding', 'entailment')."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for the alignment method."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps')."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing."
    )

    parser.add_argument(
        "--small_test",
        action="store_true",
        help="Run a small dataset for quick testing."
    )

    parser.add_argument(
        "--embedding_model_key",
        type=str,
        default="miniLM",  # or some default key
        help="Which embedding model key to use. Must be one of: miniLM, mpnet, etc."
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Initialize timer
    timer = Timer()
    timer.start()

    # 1) Create default config object
    default_config = AlignmentConfig()

    if args.embedding_model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model key: {args.embedding_model_key}. "
                         f"Must be one of {list(EMBEDDING_MODELS.keys())}")

    model_info = EMBEDDING_MODELS[args.embedding_model_key]

    # Initialize configuration with overrides if provided
    config = AlignmentConfig(
        method=args.method if args.method else default_config.method,
        threshold=args.threshold if args.threshold is not None else model_info["threshold"],
        device=check_or_select_device(args.device),
        embedding_config=EmbeddingModelConfig(
            model_name=model_info["model_name"],
            threshold=model_info["threshold"]
        ),
        cache_path=model_info["cache_file"]
    )

    # Log configuration for debugging
    print(f"Using Configuration: {config}")

    # Create the aligner based on the configuration
    aligner = create_aligner(config)

    # List of all datasets
    all_datasets = [
        DatasetName.CNNDM_TEST,
        DatasetName.CNNDM_VALIDATION,
        DatasetName.XSUM,
        DatasetName.SAMSUM,
    ]

    if args.dataset_name:
        # Process a single dataset
        if args.dataset_name not in all_datasets:
            raise ValueError(f"Unknown dataset name: {args.dataset_name}. Must be one of {all_datasets}.")
        results = process_single_dataset(args.dataset_name, aligner, small_test=args.small_test)
        save_results({args.dataset_name: results}, small_test=args.small_test)
    else:
        # Process all datasets
        combined_results = process_all_datasets(all_datasets, aligner, small_test=args.small_test)
        save_all_results(combined_results, small_test=args.small_test)

    if hasattr(aligner, "save_alignment_cache"):
        aligner.save_alignment_cache()

    timer.stop()
    print(f"Alignment processing completed in {timer.format_elapsed_time()}")


if __name__ == "__main__":
    main()
