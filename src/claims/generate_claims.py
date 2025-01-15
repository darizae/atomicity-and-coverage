import argparse

from claim_generator import ClaimGenerator
from src.config import RosePathsSmall, RosePaths, MODELS
from src.rose.rose_loader import RoseDatasetLoader

from device_selector import check_or_select_device


def get_args():
    parser = argparse.ArgumentParser(description="Generate claims from a dataset using a model.")
    parser.add_argument("--dataset_name", type=str, default="cnndm_test", help="Dataset to process.")
    parser.add_argument("--model_key", type=str, default="distilled_t5", help="Model key to use.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing claims.")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens if truncation is enabled.")
    parser.add_argument("--no_truncation", action="store_true", help="Disable truncation entirely.")
    parser.add_argument("--small_test", action="store_true", help="Use the small dataset (1 entry) for quick testing.")
    return parser.parse_args()


def main(
    dataset_name: str,
    model_key: str,
    device: str = None,
    batch_size: int = 32,
    max_length: int = 512,
    truncation: bool = True,
    small_test: bool = False,
) -> None:
    """
    Main entry point for generating claims.

    Args:
        dataset_name (str): The dataset key to process (e.g., "cnndm_test").
        model_key (str): The model key as specified in the MODELS config.
        device (str, optional): Device to run the model on (e.g., "cpu", "cuda", or "mps"). Defaults to None.
        batch_size (int, optional): Batch size for processing claims. Defaults to 32.
        max_length (int, optional): Maximum number of tokens per input sequence.
                                    Inputs longer than this will be truncated if truncation is enabled. Defaults to 512.
        truncation (bool, optional): Whether to truncate inputs that exceed `max_length`. Defaults to True.
        small_test (bool): Whether to use the small dataset (1 entry) for quick tests.

    Raises:
        KeyError: If the specified dataset is not found in the loaded datasets.
        ValueError: If an unknown model_key is provided.
    """
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
    print("Starting claim generation...")

    claims = generator.generate_claims(sources)

    print("Claim generation finished!")

    # 9. Add claims and save
    print(f"Saving claims for dataset '{dataset_name}', as '{claims_field}'.")
    loader.add_claims(dataset_name, claims_field, claims)
    loader.save_datasets_compressed(paths.compressed_dataset_with_system_claims_path)
    loader.save_datasets_json(paths.dataset_with_system_claims_path)

    print(f"Claims generated and saved for dataset '{dataset_name}' using model '{model_name}' on device '{device}'.")


if __name__ == "__main__":
    args = get_args()

    # Notice how we flip the no_truncation flag:
    truncation_flag = not args.no_truncation

    main(
        dataset_name=args.dataset_name,
        model_key=args.model_key,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncation=truncation_flag,
        small_test=args.small_test
    )
