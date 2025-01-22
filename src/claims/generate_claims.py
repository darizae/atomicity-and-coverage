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


def generate_and_save_claims_for_dataset(
    loader: RoseDatasetLoader,
    dataset_name: str,
    generator,
    model_info,
    paths
):
    """Generate and save claims for a single dataset, using an already-initialized generator."""
    if dataset_name not in loader.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in loaded datasets.")

    timer = Timer()
    timer.start()

    dataset = loader.datasets[dataset_name]
    sources = [entry["reference"] for entry in dataset]

    # Generate
    print(f"Generating claims for dataset '{dataset_name}' using model '{model_info.name}'...")
    claims = generator.generate_claims(sources)
    print("Claim generation complete.")

    # Save
    print(f"Saving claims for dataset '{dataset_name}' to field '{model_info.claims_field}'...")
    loader.add_claims(dataset_name, model_info.claims_field, claims)
    loader.save_datasets_compressed(paths.compressed_dataset_with_system_claims_path)
    loader.save_datasets_json(paths.dataset_with_system_claims_path)
    print("Save completed!")

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

    # 1. Select device once
    device = check_or_select_device(args.device)
    print(f"Using device: {device}")

    # 2. Decide paths (small vs. full) once
    paths = RosePathsSmall() if args.small_test else RosePaths()
    print(f"Using {'small' if args.small_test else 'full'} dataset paths at: {paths.compressed_dataset_path}")

    # 3. Load all datasets once
    loader = load_datasets(paths.compressed_dataset_path)

    # 4. Initialize model/generator once
    print(f"Initializing model generator for '{args.model_key}'...")
    generator, model_info = initialize_model_generator(
        model_key=args.model_key,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncation=truncation_flag,
        temperature=args.temperature,
        openai_api_key=args.openai_api_key,
    )
    print("Model generator initialized.")

    # 5. Process either one dataset or all
    if args.dataset_name:
        print(f"Processing dataset: {args.dataset_name}")
        generate_and_save_claims_for_dataset(
            loader,
            args.dataset_name,
            generator,
            model_info,
            paths
        )
    else:
        print("No specific dataset provided. Processing all known datasets...")
        for alias in DATASET_ALIASES.keys():
            print(f"Processing dataset: {alias}")
            generate_and_save_claims_for_dataset(
                loader,
                alias,
                generator,
                model_info,
                paths
            )


if __name__ == "__main__":
    main()
