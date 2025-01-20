import argparse
from pathlib import Path

from claim_generator import Seq2SeqClaimGenerator, CausalLMClaimGenerator, ModelConfig, APIClaimGenerator
from src.config import RosePathsSmall, RosePaths, CLAIM_GENERATION_MODELS, DATASET_ALIASES
from src.rose.rose_loader import RoseDatasetLoader
from src.utils.timer import Timer

from src.utils.device_selector import check_or_select_device


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


def load_datasets(compressed_path: Path) -> RoseDatasetLoader:
    loader = RoseDatasetLoader()
    print("Loading datasets...")
    loader.load_datasets_compressed(compressed_path)
    print("Datasets loaded!")
    return loader


def initialize_model_generator(
        model_key: str,
        device: str,
        batch_size: int,
        max_length: int,
        truncation: bool
):
    if model_key not in CLAIM_GENERATION_MODELS:
        raise ValueError(...)

    model_info = CLAIM_GENERATION_MODELS[model_key]

    model_type = model_info.get("type", "seq2seq")

    model_config = ModelConfig(
        model_name=model_info["name"],  # "http://127.0.0.1:1337/v1/chat/completions" or "gpt-3.5-turbo"
        tokenizer_class_path=model_info.get("tokenizer_class", ""),  # might not exist for API-based
        model_class_path=model_info.get("model_class", ""),  # might not exist for API-based
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        truncation=truncation
    )
    model_config.type = model_type  # Add a custom field if you like

    # Decide which generator to instantiate
    if model_type == "seq2seq":
        generator_cls = Seq2SeqClaimGenerator
    elif model_type == "causal":
        generator_cls = CausalLMClaimGenerator
    elif model_type in ("openai", "openai_local"):  # e.g. Jan's local server
        generator_cls = APIClaimGenerator
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")

    generator = generator_cls(model_config)
    return generator, model_info


def save_generated_claims(loader: RoseDatasetLoader, dataset_name: str, claims_field: str, claims, paths) -> None:
    print(f"Saving claims for dataset '{dataset_name}', as '{claims_field}'.")
    loader.add_claims(dataset_name, claims_field, claims)
    loader.save_datasets_compressed(paths.compressed_dataset_with_system_claims_path)
    loader.save_datasets_json(paths.dataset_with_system_claims_path)


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
    timer = Timer()
    timer.start()

    # Device selection
    device = check_or_select_device(device)
    print(f"Using device: {device}")

    # Determine dataset paths based on test size
    paths = RosePathsSmall() if small_test else RosePaths()
    print(f"Using {'small' if small_test else 'full'} test dataset path...")

    # Load datasets
    loader = load_datasets(paths.compressed_dataset_path)

    # Check dataset existence
    if dataset_name not in loader.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in loaded datasets.")

    # Initialize model and claim generator
    print("Initializing claim generator model...")
    generator, model_info = initialize_model_generator(
        model_key, device, batch_size, max_length, truncation
    )
    print("Claim generator model initialized!")

    # Retrieve sources from dataset
    dataset = loader.datasets[dataset_name]
    sources = [entry["reference"] for entry in dataset]

    # Generate claims
    print(f"Starting claim generation for dataset '{dataset_name}'...")
    claims = generator.generate_claims(sources)
    print("Claim generation finished!")

    # Save generated claims
    save_generated_claims(loader, dataset_name, model_info["claims_field"], claims, paths)

    # Log results and timing
    num_arrays = len(claims)
    num_total_claims = sum(len(claim) for claim in claims if isinstance(claim, list))
    timer.stop()

    print(f"Number of claim arrays generated: {num_arrays}")
    print(f"Total number of claims generated: {num_total_claims}")
    print(f"Time taken for dataset '{dataset_name}': {timer.format_elapsed_time()}\n")


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
        for alias in DATASET_ALIASES.keys():
            process_dataset(alias, model_key, device, batch_size, max_length, truncation, small_test)


if __name__ == "__main__":
    args = get_args()
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
