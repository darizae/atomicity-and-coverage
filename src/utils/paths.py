import re
from pathlib import Path
from dataclasses import dataclass

# Compute the base directory using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent  # Goes from config/ up to src/


@dataclass
class RosePaths:
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets.json"
    dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets.json"


@dataclass
class RosePathsSmall:
    dataset_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json"
    dataset_with_system_claims_path: Path = BASE_DIR / "rose" / "rose_datasets_small.json"


@dataclass
class AlignmentPaths:
    # Put the cache file in metrics/alignment/cache
    alignment_cache_dir: Path = BASE_DIR / "alignment" / "cache"

    # embeddings
    miniLM_cache_file: Path = alignment_cache_dir / "embedding_cache_all_MiniLM.pkl"
    mpnet_cache_file: Path = alignment_cache_dir / "embedding_cache_mpnet.pkl"

    # entailment
    roberta_mnli_cache_file: Path = alignment_cache_dir / "entailment_cache_roberta_mnli.pkl"
    bart_mnli_cache_file: Path = alignment_cache_dir / "entailment_cache_bart_mnli.pkl"


def sanitize_filename(text: str) -> str:
    """
    Makes sure any slashes or invalid filename chars are replaced.
    """
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)


def get_alignment_results_path(
        config,
        dataset_name: str = None,
        small_test: bool = False
) -> Path:
    """
    Construct a directory path and filename based on:
    - method (embedding, entailment, rouge)
    - model key (embedding or entailment)
    - claim_gen_key
    - threshold
    - dataset_name
    - small_test
    """

    base_dir = BASE_DIR / "metrics" / "alignment_results"

    # Decide which subdirectory for 'method'
    method_dir = base_dir / config.method

    # Figure out the alignment model key or fallback
    if config.method == "embedding":
        model_key = sanitize_filename(config.embedding_config.model_name)
        threshold = config.embedding_config.threshold
    elif config.method == "entailment":
        model_key = sanitize_filename(config.entailment_config.model_name)
        threshold = config.entailment_config.threshold
    elif config.method == "entailment_bipartite":
        model_key = sanitize_filename(config.entailment_config.model_name)
        threshold = config.entailment_config.threshold
    elif config.method == "embedding-bipartite":
        model_key = sanitize_filename(config.embedding_config.model_name)
        threshold = config.embedding_config.threshold
    else:
        # e.g. "rouge" has no model name
        model_key = "rouge_model"
        threshold = config.threshold  # Or skip if threshold doesn’t matter for ROUGE

    # Also consider the claim-gen key
    claim_gen_dir = sanitize_filename(config.claim_gen_key)

    # Finally, subdirectory for threshold
    threshold_dir = f"threshold_{threshold:.2f}"

    # If we want small_test in the path
    if small_test:
        test_dir = "small"
    else:
        test_dir = "full"

    # Build the full directory path
    dir_path = method_dir / model_key / claim_gen_dir / threshold_dir / test_dir
    dir_path.mkdir(parents=True, exist_ok=True)

    # Choose filename. If dataset_name is None, we might do "combined.json".
    if dataset_name:
        filename = f"{dataset_name}.json"
    else:
        filename = "combined.json"

    return dir_path / filename