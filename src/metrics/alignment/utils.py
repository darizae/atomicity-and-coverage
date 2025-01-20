import json
import os
from typing import List, Dict, Any

from src.metrics.alignment.base_aligner import BaseAligner
from src.utils.device_selector import check_or_select_device
from src.config import RosePaths, RosePathsSmall, AlignmentConfig, DatasetName
from src.config.alignment_config import EmbeddingModelConfig, EntailmentModelConfig
from src.config.models_config import EMBEDDING_MODELS, ENTAILMENT_MODELS
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage
from src.utils.path_utils import get_alignment_results_path


def build_config(args) -> AlignmentConfig:
    """
    Builds an AlignmentConfig based on the user-specified arguments and defaults.
    """
    default_config = AlignmentConfig()
    method = args.method.lower() if args.method else default_config.method

    match method:
        case "embedding":
            # Validate the model key
            if args.embedding_model_key not in EMBEDDING_MODELS:
                raise ValueError(f"Unknown embedding model key: {args.embedding_model_key}. "
                                 f"Must be one of {list(EMBEDDING_MODELS.keys())}")
            model_info = EMBEDDING_MODELS[args.embedding_model_key]

            config = AlignmentConfig(
                method=method,
                threshold=args.threshold if args.threshold is not None else model_info["threshold"],
                device=check_or_select_device(args.device),
                embedding_config=EmbeddingModelConfig(
                    model_name=model_info["model_name"],
                    threshold=model_info["threshold"]
                ),
                cache_path=model_info["cache_file"]
            )

        case "entailment":
            # Validate the model key
            if args.entailment_model_key not in ENTAILMENT_MODELS:
                raise ValueError(f"Unknown entailment model key: {args.entailment_model_key}. "
                                 f"Must be one of {list(ENTAILMENT_MODELS.keys())}")
            model_info = ENTAILMENT_MODELS[args.entailment_model_key]

            config = AlignmentConfig(
                method=method,
                threshold=args.threshold if args.threshold is not None else model_info["threshold"],
                device=check_or_select_device(args.device),
                entailment_config=EntailmentModelConfig(
                    model_name=model_info["model_name"],
                    threshold=model_info["threshold"]
                ),
                cache_path=model_info["cache_file"]
            )

        case "rouge":
            # Possibly ignore threshold or device?
            config = AlignmentConfig(
                method=method,
                threshold=args.threshold if args.threshold is not None else default_config.threshold,
                device=check_or_select_device(args.device),
            )

        case _:
            # Fallback if user typed a method not recognized
            # or you can treat it as 'rouge' or throw an error
            raise ValueError(f"Unknown method: {method}. Options: 'embedding', 'entailment', 'rouge'.")

    claim_gen_key = args.claim_gen_key or default_config.claim_gen_key
    config.claim_gen_key = claim_gen_key

    return config


def do_alignment(
    dataset_name: str,
    aligner: BaseAligner,
    small_test: bool,
    config: AlignmentConfig
) -> None:
    """
    Handles either a single dataset or all datasets. If dataset_name is provided, we process that
    specific dataset; otherwise we process them all.
    """
    # The list of all datasets
    all_datasets = [
        DatasetName.CNNDM_TEST,
        DatasetName.CNNDM_VALIDATION,
        DatasetName.XSUM,
        DatasetName.SAMSUM,
    ]

    if dataset_name:
        # Single dataset
        if dataset_name not in all_datasets:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. "
                f"Must be one of {all_datasets}."
            )
        results = process_single_dataset(
            dataset_name,
            aligner,
            small_test=small_test
        )
        save_results(config, results, dataset_name, small_test=small_test)
    else:
        # All datasets
        combined_results = process_all_datasets(
            all_datasets,
            aligner,
            small_test=small_test
        )
        save_all_results(config, combined_results, small_test=small_test)

        # Save alignment cache if supported
    if hasattr(aligner, "save_alignment_cache"):
        aligner.save_alignment_cache()


def _load_dataset(small_test: bool = False):
    # Determine the path based on dataset subset
    paths = RosePathsSmall() if small_test else RosePaths()
    dataset_path = paths.compressed_dataset_path

    loader = RoseDatasetLoader()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
    loader.load_datasets_compressed(dataset_path)

    return loader.datasets if not small_test else {k: v[:1] for k, v in loader.datasets.items()}


def _process_dataset(dataset, aligner):
    results = []
    for record in dataset:
        system_claims = record.get("system_claims_t5", [])
        reference_acus = record.get("reference_acus", [])
        if not system_claims or not reference_acus:
            continue

        alignment_map = aligner.align(system_claims, reference_acus)
        coverage = compute_coverage(alignment_map, len(reference_acus))
        atomicity = compute_atomicity(alignment_map, len(system_claims))

        results.append({
            "reference summary": record.get("reference", ""),
            "coverage": coverage,
            "atomicity": atomicity,
            "alignment_map": alignment_map
        })
    return results


def process_all_datasets(
        datasets: List[str],
        aligner,
        small_test: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all datasets and aggregate results into a single structure.

    Args:
        datasets (List[str]): List of dataset names to process.
        aligner: The alignment model.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Combined results from all datasets.
    """
    combined_results = {}
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
        results = _process_dataset(dataset, aligner)
        combined_results[dataset_name] = results
        print(f"Finished processing dataset: {dataset_name}")
    return combined_results


def process_single_dataset(
        dataset_name: str,
        aligner,
        small_test: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a single dataset.

    Args:
        dataset_name (str): Name of the dataset to process.
        aligner: The alignment model.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        List[Dict[str, Any]]: Results for the single dataset.
    """
    print(f"Processing single dataset: {dataset_name}")
    dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
    return _process_dataset(dataset, aligner)


def save_results(
        config,
        results: List[Dict[str, Any]],
        dataset_name: str,
        small_test: bool = False
) -> None:
    """
    Save the results to a JSON file constructed by the experiment config & dataset name.
    """
    output_path = get_alignment_results_path(
        config=config,
        dataset_name=dataset_name,
        small_test=small_test
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[save_results] Results saved to {output_path}")


def save_all_results(
        config,
        all_results: Dict[str, List[Dict[str, Any]]],
        small_test: bool = False
) -> None:
    """
    Save all dataset results to a single 'combined.json' path for the entire run config.
    """
    output_path = get_alignment_results_path(
        config=config,
        dataset_name=None,  # combined
        small_test=small_test
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[save_all_results] Combined results saved to {output_path}")
