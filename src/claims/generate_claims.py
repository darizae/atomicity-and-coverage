import argparse
from pathlib import Path

from src.claims.claim_generator import ModelConfig
from src.claims.config import get_claim_generation_model_config
from src.metrics.datasets_config import DATASET_ALIASES
from src.utils.paths import RosePathsSmall, RosePaths
from src.rose.rose_loader import RoseDatasetLoader
from src.utils.timer import Timer

from src.utils.device_selector import check_or_select_device


def get_args():
    parser = argparse.ArgumentParser(description="Generate claims from datasets using a model.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset to process (or leave empty for all).")
    parser.add_argument("--model_key", type=str, default="distilled_t5",
                        help="Which model (key in CLAIM_GENERATION_MODELS) to use.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g. 'cuda', 'cpu', 'mps'). If not provided, auto-select.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum input tokens (if truncation is on).")
    parser.add_argument("--no_truncation", action="store_true",
                        help="Disable truncation entirely.")
    parser.add_argument("--small_test", action="store_true",
                        help="Use the small dataset for quick testing.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (for all providers).")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="Optional explicit OpenAI API key.")
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
    truncation: bool,
    temperature: float,
    openai_api_key: str = None,
):
    """
    Given a model_key from CLAIM_GENERATION_MODELS, build a ModelConfig
    and instantiate the appropriate generator class.
    """
    model_info = get_claim_generation_model_config(model_key)

    config = ModelConfig(
        model_name_or_path=model_info.name,
        tokenizer_class=model_info.tokenizer_class,
        model_class=model_info.model_class,
        endpoint_url=model_info.endpoint_url,
        openai_api_key=openai_api_key,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        truncation=truncation,
        temperature=temperature,
    )

    generator_class = model_info.generator_cls
    generator = generator_class(config)

    return generator, model_info


def save_generated_claims(loader: RoseDatasetLoader, dataset_name: str,
                          claims_field: str, claims, paths) -> None:
    """
    Insert the newly generated claims into the dataset, then
    save (both compressed and JSON).
    """
    print(f"Saving claims for dataset '{dataset_name}' to field '{claims_field}'...")
    loader.add_claims(dataset_name, claims_field, claims)
    loader.save_datasets_compressed(paths.compressed_dataset_with_system_claims_path)
    loader.save_datasets_json(paths.dataset_with_system_claims_path)
    print("Save completed!")


def process_dataset(
    dataset_name: str,
    model_key: str,
    device: str,
    batch_size: int,
    max_length: int,
    truncation: bool,
    small_test: bool,
    temperature: float,
    openai_api_key: str = None,
) -> None:
    """Generate claims for a single dataset using a specified model."""
    timer = Timer()
    timer.start()

    # Device selection
    device = check_or_select_device(device)
    print(f"Using device: {device}")

    # Pick which dataset paths (small or full)
    paths = RosePathsSmall() if small_test else RosePaths()
    print(f"Using {'small' if small_test else 'full'} dataset paths at: {paths.compressed_dataset_path}")

    # Load
    loader = load_datasets(paths.compressed_dataset_path)
    if dataset_name not in loader.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in loaded datasets.")

    # Initialize model/generator
    print(f"Initializing model generator for '{model_key}'...")
    generator, model_info = initialize_model_generator(
        model_key=model_key,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        truncation=truncation,
        temperature=temperature,
        openai_api_key=openai_api_key,
    )
    print("Model generator initialized.")

    # Get input texts
    dataset = loader.datasets[dataset_name]
    sources = [entry["reference"] for entry in dataset]

    # Generate
    print(f"Generating claims for dataset '{dataset_name}'...")
    claims = generator.generate_claims(sources)
    print("Claim generation complete.")

    # Save
    save_generated_claims(loader, dataset_name, model_info.claims_field, claims, paths)

    # Log
    num_arrays = len(claims)
    num_total_claims = sum(len(c) for c in claims if isinstance(c, list))
    timer.stop()

    print(f"Number of claim arrays generated: {num_arrays}")
    print(f"Total number of claims generated: {num_total_claims}")
    print(f"Time taken: {timer.format_elapsed_time()}\n")


def main():
    args = get_args()
    truncation_flag = not args.no_truncation

    if args.dataset_name:
        print(f"Processing dataset: {args.dataset_name}")
        process_dataset(
            dataset_name=args.dataset_name,
            model_key=args.model_key,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            truncation=truncation_flag,
            small_test=args.small_test,
            temperature=args.temperature,
            openai_api_key=args.openai_api_key,
        )
    else:
        print("No specific dataset provided. Processing all known datasets...")
        for alias in DATASET_ALIASES.keys():
            process_dataset(
                dataset_name=alias,
                model_key=args.model_key,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                truncation=truncation_flag,
                small_test=args.small_test,
                temperature=args.temperature,
                openai_api_key=args.openai_api_key,
            )


if __name__ == "__main__":
    main()
