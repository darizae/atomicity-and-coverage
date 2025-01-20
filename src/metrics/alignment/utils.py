import json
import os
from typing import List, Dict, Any

from src.metrics.alignment.base_aligner import BaseAligner
from src.utils.device_selector import check_or_select_device
from src.config import RosePaths, RosePathsSmall, AlignmentConfig, DatasetName, DATASET_ALIASES
from src.config.alignment_config import EmbeddingModelConfig, EntailmentModelConfig, AlignmentMethods
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage
from src.utils.path_utils import get_alignment_results_path

from src.config.models_config import EMBEDDING_MODELS, ENTAILMENT_MODELS, CLAIM_GENERATION_MODELS


def build_config(args) -> AlignmentConfig:
    """
    Build an AlignmentConfig based on user-specified arguments (args) and defaults.
    """
    default_config = AlignmentConfig()

    method = (args.method or default_config.method).lower()
    raw_dataset_name = args.dataset_name or default_config.dataset_name
    dataset_name = DATASET_ALIASES.get(raw_dataset_name, raw_dataset_name)
    device = check_or_select_device(args.device)
    claim_gen_key = args.claim_gen_key or default_config.claim_gen_key

    if method == AlignmentMethods.EMBEDDING:
        if args.embedding_model_key not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding model key: {args.embedding_model_key}. "
                f"Must be one of {list(EMBEDDING_MODELS.keys())}"
            )
        model_info = EMBEDDING_MODELS[args.embedding_model_key]
        threshold = args.threshold if args.threshold is not None else model_info["threshold"]

        config = AlignmentConfig(
            method=method,
            threshold=threshold,
            device=device,
            embedding_config=EmbeddingModelConfig(
                model_name=model_info["model_name"],
                threshold=model_info["threshold"],
            ),
            cache_path=model_info["cache_file"],
            claim_gen_key=claim_gen_key,
            dataset_name=dataset_name
        )

    elif method == AlignmentMethods.ENTAILMENT:
        if args.entailment_model_key not in ENTAILMENT_MODELS:
            raise ValueError(
                f"Unknown entailment model key: {args.entailment_model_key}. "
                f"Must be one of {list(ENTAILMENT_MODELS.keys())}"
            )
        model_info = ENTAILMENT_MODELS[args.entailment_model_key]
        threshold = args.threshold if args.threshold is not None else model_info["threshold"]

        config = AlignmentConfig(
            method=method,
            threshold=threshold,
            device=device,
            entailment_config=EntailmentModelConfig(
                model_name=model_info["model_name"],
                threshold=model_info["threshold"],
            ),
            cache_path=model_info["cache_file"],
            claim_gen_key=claim_gen_key,
            dataset_name=dataset_name
        )

    elif method == AlignmentMethods.ROUGE:
        threshold = args.threshold if args.threshold is not None else default_config.threshold

        config = AlignmentConfig(
            method=method,
            threshold=threshold,
            device=device,
            claim_gen_key=claim_gen_key,
            dataset_name=dataset_name
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Options: '{AlignmentMethods.EMBEDDING}', '{AlignmentMethods.ENTAILMENT}', '{AlignmentMethods.ROUGE}'."
        )

    return config


def do_alignment(
        aligner: BaseAligner,
        small_test: bool,
        config: AlignmentConfig
) -> None:
    """
    Handles either a single dataset or all datasets. If dataset_name is provided, we process that
    specific dataset; otherwise we process them all.
    """
    all_datasets = [
        DatasetName.CNNDM_TEST,
        DatasetName.CNNDM_VALIDATION,
        DatasetName.XSUM,
        DatasetName.SAMSUM,
    ]
    dataset_name = config.dataset_name

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
            config,
            small_test=small_test
        )
        save_results(config, results, dataset_name, small_test=small_test)
    else:
        # All datasets
        combined_results = process_all_datasets(
            all_datasets,
            aligner,
            config,
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


def _process_dataset(dataset, aligner, config):
    results = []

    system_claims_key = CLAIM_GENERATION_MODELS[config.claim_gen_key]["claims_field"]

    for record in dataset:
        # Grab system claims using the dynamic key
        system_claims = record.get(system_claims_key, [])
        reference_acus = record.get("reference_acus", [])

        if not system_claims or not reference_acus:
            # If either is empty, skip this record
            continue

        # 1) Get the raw alignment map (indexes only)
        alignment_map = aligner.align(system_claims, reference_acus)

        # 2) Expand it to a human-readable structure
        alignment_map_expanded = expand_alignment_map(system_claims, reference_acus, alignment_map)

        # 3) Compute coverage & atomicity from the raw alignment_map
        coverage = compute_coverage(alignment_map, len(reference_acus))
        atomicity = compute_atomicity(alignment_map, len(system_claims))

        # 4) Save final record results
        results.append({
            "reference summary": record.get("reference", ""),
            "coverage": coverage,
            "atomicity": atomicity,
            "alignment_map_expanded": alignment_map_expanded
        })

    return results


def process_all_datasets(
        datasets: List[str],
        aligner,
        config: AlignmentConfig,
        small_test: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all datasets and aggregate results into a single structure.

    Args:
        datasets (List[str]): List of dataset names to process.
        aligner: The alignment model.
        config: AlignmentConfig instance.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Combined results from all datasets.
    """
    combined_results = {}
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
        results = _process_dataset(dataset, aligner, config)
        combined_results[dataset_name] = results
        print(f"Finished processing dataset: {dataset_name}")
    return combined_results


def process_single_dataset(
        dataset_name: str,
        aligner,
        config: AlignmentConfig,
        small_test: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a single dataset.

    Args:
        dataset_name (str): Name of the dataset to process.
        aligner: The alignment model.
        config: AlignmentConfig instance.
        small_test (bool): Whether to use the small dataset variant.

    Returns:
        List[Dict[str, Any]]: Results for the single dataset.
    """
    print(f"Processing single dataset: {dataset_name}")
    dataset = _load_dataset(small_test=small_test).get(dataset_name, [])
    return _process_dataset(dataset, aligner, config)


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


def expand_alignment_map(
    system_claims: List[str],
    reference_acus: List[str],
    alignment_map: Dict[int, List[int]]
) -> List[Dict[str, Any]]:
    """
    Convert the raw alignment map {system_idx: [ref_idx, ...]}
    into a list of dicts with both text and indexes.
    """
    expanded = []
    for s_idx, matched_ref_idxs in alignment_map.items():
        sys_text = system_claims[s_idx]
        matched_refs = []
        for r_idx in matched_ref_idxs:
            matched_refs.append({
                "ref_idx": r_idx,
                "ref_claim": reference_acus[r_idx]
            })
        expanded.append({
            "system_claim_idx": s_idx,
            "system_claim_text": sys_text,
            "matched_refs": matched_refs
        })
    return expanded

