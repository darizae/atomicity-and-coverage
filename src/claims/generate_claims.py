import argparse
import time
from typing import Tuple

from claim_generator import ClaimGenerator
from src.config import RosePathsSmall, RosePaths, MODELS, DATASET_ALIASES
from src.rose.rose_loader import RoseDatasetLoader

from device_selector import check_or_select_device


def get_args():
    parser = argparse.ArgumentParser(description="Generate claims from datasets using a model.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset to process (or leave empty for all).")
    parser.add_argument("--model_key", type=str, default="distilled_t5", help="Model key to use.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing claims.")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens if truncation is enabled.")
    parser.add_argument("--no_truncation", action="store_true", help="Disable truncation entirely.")
    parser.add_argument("--small_test", action="store_true", help="Use the small dataset (1 entry) for quick testing.")
    return parser.parse_args()


def process_dataset(
        dataset_name: str,
        model_key: str,
        device: str,
        batch_size: int,
        max_length: int,
        truncation: bool,
        small_test: bool,
) -> None:
    """Processes a single dataset and generates claims."""
    start_time = time.time()

    # 1. Determine device
    device = check_or_select_device(device)
    print(f"Using device: {device}")

    # 2. Initialize paths
    print("Determining dataset variant to use...")
    if small_test:
        paths = RosePathsSmall()
        print("Using small test dataset path...")
    else:
        paths = RosePaths()
        print("Using full test dataset path...")

    # 3. Load dataset
    loader = RoseDatasetLoader()

    print("Loading datasets...")
    loader.load_datasets_compressed(paths.compressed_dataset_path)
    print("Datasets loaded!")

    # 4. Check if dataset exists
    if dataset_name not in loader.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in loaded datasets.")

    # 5. Retrieve model info
    print("Initializing claim generator model...")

    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. "
                         f"Supported keys: {list(MODELS.keys())}")

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    claims_field = model_info["claims_field"]

    # If you provided custom classes:
    tokenizer_class = model_info.get("tokenizer_class", "transformers.AutoTokenizer")
    model_class = model_info.get("model_class", "transformers.AutoModelForSeq2SeqLM")

    # 6. Initialize claim generator
    generator = ClaimGenerator(
        model_name=model_name,
        tokenizer_class_path=tokenizer_class,
        model_class_path=model_class,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        truncation=truncation
    )

    print("Claim generator model initialized!")

    # 7. Retrieve sources
    dataset = loader.datasets[dataset_name]
    sources = [entry["reference"] for entry in dataset]

    # 8. Generate claims
    print(f"Starting claim generation for dataset '{dataset_name}'...")

    claims = generator.generate_claims(sources)

    print("Claim generation finished!")

    # 9. Add claims and save
    print(f"Saving claims for dataset '{dataset_name}', as '{claims_field}'.")
    loader.add_claims(dataset_name, claims_field, claims)
    loader.save_datasets_compressed(paths.compressed_dataset_with_system_claims_path)
    loader.save_datasets_json(paths.dataset_with_system_claims_path)

    # Log the number of claim arrays and total claims
    num_arrays = len(claims)
    num_total_claims = sum(
        len(claim) for claim in claims if isinstance(claim, list))  # Assumes claims is a list of lists

    end_time = time.time()
    hours, minutes, seconds = _compute_elapsed_time(start_time, end_time)

    print(f"Number of claim arrays generated: {num_arrays}")
    print(f"Total number of claims generated: {num_total_claims}")
    print(f"Time taken for dataset '{dataset_name}': {hours} hours, {minutes} minutes, and {seconds:.2f} seconds\n")


def main(
        model_key: str,
        device: str,
        batch_size: int,
        max_length: int,
        truncation: bool,
        small_test: bool,
        dataset_name: str = None,
) -> None:
    """
    Main function to process one or all datasets based on arguments.

    Args:
        model_key (str): Model key to use.
        device (str): Device to run the model on.
        batch_size (int): Batch size for processing claims.
        max_length (int): Maximum number of tokens per input sequence.
        truncation (bool): Whether to truncate inputs that exceed `max_length`.
        small_test (bool): Whether to use the small dataset (1 entry) for quick tests.
        dataset_name (str, optional): Process a specific dataset. If None, process all datasets in DATASETS_CONFIG.
    """
    if dataset_name:
        print(f"Processing dataset: {dataset_name}")
        process_dataset(dataset_name, model_key, device, batch_size, max_length, truncation, small_test)
    else:
        print("Processing all datasets...")
        for alias, hf_name in DATASET_ALIASES.items():
            process_dataset(alias, model_key, device, batch_size, max_length, truncation, small_test)


def _compute_elapsed_time(start_time: float, end_time: float) -> tuple[int, int, float]:
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return hours, minutes, seconds


if __name__ == "__main__":
    args = get_args()

    # Notice how we flip the no_truncation flag:
    truncation_flag = not args.no_truncation

    main(
        model_key=args.model_key,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncation=truncation_flag,
        small_test=args.small_test,
        dataset_name=args.dataset_name,
    )
