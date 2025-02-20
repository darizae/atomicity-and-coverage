import argparse

from my_timer import Timer

from src.alignment.alignment_factory import create_aligner
from src.alignment.utils import build_config, do_alignment


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
        default="miniLM",
        help="Which embedding model key to use. Must be one of: miniLM, mpnet, etc."
    )

    parser.add_argument(
        "--entailment_model_key",
        type=str,
        default="roberta",
        help="Which entailment model key to use. Must be one of: roberta, bart, etc."
    )

    parser.add_argument(
        "--claim_gen_key",
        type=str,
        default="distilled_t5",
        help="Which claims from which claim generation model to use. Must be one of: distilled_t5, etc."
    )

    parser.add_argument(
        "--reference_claims_key",
        type=str,
        default="reference_acus_deduped_0.9_select_longest",
        help="Which claims from which claim generation model to use. Default is best deduped RoSE."
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Initialize timer
    timer = Timer()
    timer.start()

    # Build the alignment config
    config = build_config(args)

    # Log configuration for debugging
    print(f"Using Configuration: {config}")

    # Create the aligner based on the configuration
    aligner = create_aligner(config)

    do_alignment(
        aligner=aligner,
        small_test=args.small_test,
        config=config
    )

    timer.stop()
    timer.print_elapsed_time()


if __name__ == "__main__":
    main()
