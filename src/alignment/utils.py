import json
import os
from typing import List, Dict, Any

from device_selector import check_or_select_device

from src.alignment.base_aligner import BaseAligner
from src.alignment.config import AlignmentConfig, AlignmentMethods, get_embedding_model_definition, \
    EmbeddingModelConfig, get_entailment_model_definition, EntailmentModelConfig
from src.metrics.datasets_config import DATASET_ALIASES, DatasetName
from src.utils.paths import RosePathsSmall, RosePaths, get_alignment_results_path
from src.rose.rose_loader import RoseDatasetLoader
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage


def build_config(args) -> AlignmentConfig:
    default_config = AlignmentConfig()

    method = (args.method or default_config.method).lower()
    raw_dataset_name = args.dataset_name or default_config.dataset_name
    dataset_name = DATASET_ALIASES.get(raw_dataset_name, raw_dataset_name)
    device = check_or_select_device(args.device)
    claim_gen_key = args.claim_gen_key or default_config.claim_gen_key
    threshold = args.threshold if args.threshold is not None else default_config.threshold

    config_params = {
        "method": method,
        "threshold": threshold,
        "device": device,
        "claim_gen_key": claim_gen_key,
        "dataset_name": dataset_name,
        "dedup_threshold": args.dedup_threshold,
        "dedup_strategy": args.dedup_strategy
    }

    if method == AlignmentMethods.EMBEDDING:
        definition = get_embedding_model_definition(args.embedding_model_key)
        threshold = args.threshold if args.threshold is not None else definition.threshold

        config_params.update({
            "threshold": threshold,
            "embedding_config": EmbeddingModelConfig(
                model_name=definition.model_name,
                cache_file=definition.cache_file,
                threshold=threshold,
            ),
            "cache_path": definition.cache_file
        })

    elif method == AlignmentMethods.ENTAILMENT:
        definition = get_entailment_model_definition(args.entailment_model_key)
        threshold = args.threshold if args.threshold is not None else definition.threshold

        config_params.update({
            "threshold": threshold,
            "entailment_config": EntailmentModelConfig(
                model_name=definition.model_name,
                cache_file=definition.cache_file,
                threshold=threshold,
            ),
            "cache_path": definition.cache_file
        })

    elif method == AlignmentMethods.ROUGE:
        pass

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Options: '{AlignmentMethods.EMBEDDING}', '{AlignmentMethods.ENTAILMENT}', '{AlignmentMethods.ROUGE}'."
        )

    return AlignmentConfig(**config_params)


def do_alignment(
        aligner: BaseAligner,
        small_test: bool,
        config: AlignmentConfig,
        dedup_threshold: float = None,
        dedup_strategy: str = None
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
            small_test=small_test,
            dedup_threshold=dedup_threshold,
            dedup_strategy=dedup_strategy
        )
        save_results(config, results, dataset_name, small_test=small_test)
    else:
        # All datasets
        combined_results = process_all_datasets(
            all_datasets,
            aligner,
            config,
            small_test=small_test,
            dedup_threshold=dedup_threshold,
            dedup_strategy=dedup_strategy
        )
        save_all_results(config, combined_results, small_test=small_test)

        # Save alignment cache if supported
    if hasattr(aligner, "save_alignment_cache"):
        aligner.save_alignment_cache()


def _load_dataset(small_test: bool = False):
    # Determine the path based on dataset subset
    paths = RosePathsSmall() if small_test else RosePaths()
    dataset_path = paths.dataset_path

    loader = RoseDatasetLoader()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
    loader.load_datasets_json(dataset_path)

    return loader.datasets if not small_test else {k: v[:1] for k, v in loader.datasets.items()}


def _process_dataset(
        dataset,
        aligner,
        config,
        dedup_threshold: float = None,
        dedup_strategy: str = None,
):
    """
    :param dedup_threshold: e.g. 0.3
    :param dedup_strategy: e.g. "select_longest"
    :returns: a list of results dict
    """

    dedup_key = ""
    results = []
    claim_gen_key = config.claim_gen_key  # e.g. "distilled_t5"

    for record in dataset:
        system_claims = (
            record.get("system_claims", {}).get(claim_gen_key, [])
        )

        if dedup_threshold is not None and dedup_strategy is not None:
            # Construct the correct key, e.g. "deduped_0.3_select_longest"
            dedup_key = f"deduped_{dedup_threshold}_{dedup_strategy}"
            reference_acus = record.get("reference_acus", {}).get(dedup_key, [])
        else:
            # Use original references by default
            reference_acus = record.get("reference_acus", {}).get("original", [])

        if not system_claims or not reference_acus:
            continue

        alignment_map = aligner.align(system_claims, reference_acus)
        coverage = compute_coverage(alignment_map, len(reference_acus))
        atomicity = compute_atomicity(alignment_map, len(system_claims))

        results.append({
            "record_id": record.get("record_id", ""),
            "coverage": coverage,
            "atomicity": atomicity,
            "reference_key": dedup_key if dedup_threshold else "original"
        })

    return results


def process_all_datasets(
        datasets: List[str],
        aligner,
        config: AlignmentConfig,
        small_test: bool = False,
        dedup_threshold: float = None,
        dedup_strategy: str = None
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
        results = _process_dataset(dataset, aligner, config, dedup_threshold, dedup_strategy)
        combined_results[dataset_name] = results
        print(f"Finished processing dataset: {dataset_name}")
    return combined_results


def process_single_dataset(
        dataset_name: str,
        aligner,
        config: AlignmentConfig,
        small_test: bool = False,
        dedup_threshold: float = None,
        dedup_strategy: str = None
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
    results = _process_dataset(dataset, aligner, config, dedup_threshold, dedup_strategy)
    return results


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
