import argparse
import torch

from claim_generator import ClaimGenerator
from src.datasets.rose_loader import RoseDatasetLoader

from config import RosePaths, MODELS
from device_selector import select_best_device


def main(dataset_name: str, model_key: str, device: str = None) -> None:
    """
    Main entry point to generate claims for a chosen dataset using a chosen model.
    :param dataset_name: The dataset key to process (e.g., "cnndm_test").
    :param model_key: The model key (e.g., "distilled_t5").
    :param device: Optional. If provided, use this device; otherwise auto-detect.
    """
    # 1. Determine device
    if device is None:
        device = select_best_device()
    print(f"Using device: {device}")

    # 2. Initialize paths
    paths = RosePaths()

    # 3. Load dataset
    loader = RoseDatasetLoader()
    loader.load_datasets_compressed(paths.dataset_path)

    # 4. Check if dataset exists
    if dataset_name not in loader.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in loaded datasets.")

    # 5. Retrieve model info
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. "
                         f"Supported keys: {list(MODELS.keys())}")

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    claims_field = model_info["claims_field"]

    # 6. Initialize claim generator
    generator = ClaimGenerator(model_name=model_name, device=device)

    # 7. Retrieve sources
    dataset = loader.datasets[dataset_name]
    sources = [entry["reference"] for entry in dataset]

    # 8. Generate claims
    claims = generator.generate_claims(sources)

    # 9. Add claims and save
    loader.add_claims(dataset_name, claims_field, claims)
    loader.save_datasets_compressed(paths.output_path)

    print(f"Claims generated and saved for dataset '{dataset_name}' using model '{model_name}' on device '{device}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate claims from a dataset using a T5 model.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnndm_test",
        help="Name of the dataset to process (e.g. 'cnndm_test')."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="distilled_t5",
        help="Key of the model to use (e.g. 'distilled_t5')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to use (e.g. 'cuda', 'cpu', 'mps'). If not provided, the script "
            "will auto-detect the best available device."
        )
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        model_key=args.model_key,
        device=args.device
    )
