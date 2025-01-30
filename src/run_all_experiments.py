"""
run_all_experiments.py

Run *all* coverage & atomicity experiments in a single pass, reusing/caching models
to avoid repeated overhead of 810 separate runs.

Usage:
  python run_all_experiments.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from my_timer import Timer

# Import your existing code
from src.alignment.config import AlignmentConfig, AlignmentMethods
from src.alignment.embedding_alignment import EmbeddingAligner
from src.alignment.entailment_alignment import EntailmentAligner
from src.alignment.rouge_alignment import RougeAligner
from src.alignment.utils import expand_alignment_map
from src.metrics.atomicity_coverage import compute_atomicity, compute_coverage
from src.rose.rose_loader import RoseDatasetLoader
from src.utils.paths import RosePathsSmall, RosePaths

# ------------------------------------------------------------------------------
# 1. Define your experiment “grid” here
# ------------------------------------------------------------------------------
METHODS = ["rouge", "embedding", "entailment"]
# METHODS = ["embedding"]
ALIGN_THRESH_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ALIGN_THRESH_LIST = [0.9]
REF_DEDUP_THRESHOLDS = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# REF_DEDUP_THRESHOLDS = [0.9]
REF_DEDUP_STRATEGIES = ["select_longest"]  # or more if needed
CLAIM_GEN_MODELS = ["distilled_t5", "gpt-3.5-turbo", "llama2_7b"]
# CLAIM_GEN_MODELS = ["llama2_7b"]

# Optionally restrict datasets or do them all
DATASET_NAMES = ["cnndm_test", "cnndm_validation", "xsum", "samsum"]
# or just do a single dataset for debugging:
# DATASET_NAMES = ["cnndm_test"]

USE_SMALL_TEST = False  # Toggle to True if you want the small version


# ------------------------------------------------------------------------------
# 2. Helpers to load your data once
# ------------------------------------------------------------------------------
def load_rose_dataset(small: bool):
    if small:
        paths = RosePathsSmall()
    else:
        paths = RosePaths()
    loader = RoseDatasetLoader()
    loader.load_datasets_json(paths.dataset_path)
    return loader.datasets


# ------------------------------------------------------------------------------
# 3. Single function to get a “model aligner” once per method
#    We *fix* device, etc. Or make them arguments.
# ------------------------------------------------------------------------------
def build_aligner(method: str, device: str = "mps"):
    if method == AlignmentMethods.EMBEDDING:
        # You might want to override threshold here with something minimal,
        # because we will do "post-filtering" by threshold anyway.
        # So we can set threshold=0.0 to get raw scores, then filter manually.
        aligner = EmbeddingAligner(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            threshold=0.0,  # we'll do thresholding ourselves
            device=device,
            cache_path="src/alignment/cache/embedding_cache_all_MiniLM.pkl"
        )

    elif method == AlignmentMethods.ENTAILMENT:
        # same logic, threshold=0.0 to get raw probabilities
        aligner = EntailmentAligner(
            model_name="roberta-large-mnli",
            threshold=0.0,  # manual threshold
            device=device,
            cache_path="src/alignment/cache/entailment_cache_roberta_mnli.pkl"
        )

    elif method == AlignmentMethods.ROUGE:
        # For ROUGE, we either do the same approach or just set threshold=0.0.
        # Because we'll do post-processing of the F1 scores ourselves.
        # If you prefer, you can keep the Aligner code as-is but
        # we can also replicate the code to compute the raw rouge1_f.
        aligner = RougeAligner(threshold=0.0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return aligner


# ------------------------------------------------------------------------------
# 4. Main function that does “all experiments” for a single method & single dataset
# ------------------------------------------------------------------------------
def run_experiments_for_dataset(dataset_name: str, method: str, all_data: Dict[str, List[dict]]):
    """
    all_data is the entire dict from load_rose_dataset(...),
    so all_data[dataset_name] is a list of records.

    We do the alignment in two steps:
      (A) For each record, for the relevant system-claims vs. each dedup'ed reference,
          compute or retrieve the *raw score* (embedding similarity, NLI prob, or ROUGE-F).
      (B) For each threshold T, do coverage & atomicity from the stored scores.
    """
    # 4.1 Build the aligner with threshold=0.0
    aligner = build_aligner(method)

    records = all_data.get(dataset_name, [])
    if not records:
        print(f"No data for dataset {dataset_name}")
        return []

    results_all = []  # we will store a big list of results objects

    for claim_gen_key in CLAIM_GEN_MODELS:
        print(f"  >> Claim-Generation Model: {claim_gen_key}")
        for dedup_t in REF_DEDUP_THRESHOLDS:
            for dedup_strat in REF_DEDUP_STRATEGIES:

                # Step (A) For each record, we gather all system claims vs. references
                # But references differ if dedup_t is not None
                # We'll store the raw alignment scores in a structure:
                #    alignment_scores_map[record_id][sys_idx][ref_idx] = raw_score
                alignment_scores_map = {}

                dedup_key = None
                if dedup_t is not None:
                    dedup_key = f"deduped_{dedup_t}_{dedup_strat}"

                # -- 4.1.1: Build a list of (system_claims, reference_acus) for each record
                #           Then do alignment *once* for all pairs to get raw scores
                #           We do this record-by-record so we do not explode memory usage.
                for record in records:
                    record_id = record.get("record_id", "")
                    system_claims = record.get("system_claims", {}).get(claim_gen_key, [])
                    if dedup_key is not None:
                        reference_acus = record.get("reference_acus", {}).get(dedup_key, [])
                    else:
                        reference_acus = record.get("reference_acus", {}).get("original", [])

                    if not system_claims or not reference_acus:
                        # skip empty
                        continue

                    # Instead of using aligner.align(... threshold=0.0),
                    # we can replicate the logic to retrieve raw scores
                    # (esp. for ROUGE or embedding), or simply patch the aligner to store raw scores.
                    # For clarity, let's do a simple approach: set aligner.threshold= -1.0 temporarily,
                    # then let aligner return "all pairs." We’ll store the score inside aligner somehow.
                    # Or we can do a custom function: get_raw_scores(...).
                    raw_scores = get_raw_alignment_scores(aligner, system_claims, reference_acus, method)

                    # Store in alignment_scores_map
                    alignment_scores_map[record_id] = raw_scores

                # Step (B) Now that we have all raw scores, we run each align_threshold in
                # ALIGN_THRESH_LIST, filter pairs, compute coverage & atomicity, then record results.

                for align_t in ALIGN_THRESH_LIST:
                    # We gather coverage & atomicity for each record, then average or store them
                    results_for_this_config = []
                    for record in records:
                        record_id = record.get("record_id", "")
                        if record_id not in alignment_scores_map:
                            # means we had no claims or references
                            continue
                        raw_scores = alignment_scores_map[record_id]

                        # raw_scores is a 2D array [sys_idx][ref_idx] = float
                        system_claims = record.get("system_claims", {}).get(claim_gen_key, [])
                        if dedup_key is not None:
                            reference_acus = record.get("reference_acus", {}).get(dedup_key, [])
                        else:
                            reference_acus = record.get("reference_acus", {}).get("original", [])

                        # build alignment_map = {sys_idx: [ref_idx,...], ...}
                        alignment_map = {}
                        for s_idx in range(len(system_claims)):
                            alignment_map[s_idx] = []
                            for r_idx in range(len(reference_acus)):
                                score = raw_scores[s_idx][r_idx]
                                if score >= align_t:
                                    alignment_map[s_idx].append(r_idx)

                        coverage = compute_coverage(alignment_map, len(reference_acus))
                        atomicity = compute_atomicity(alignment_map, len(system_claims))

                        results_for_this_config.append({
                            "record_id": record_id,
                            "coverage": coverage,
                            "atomicity": atomicity
                        })

                    # You can choose to average or just store raw. Let’s just store them:
                    # e.g. an entry for each record:
                    config_result = {
                        "method": method,
                        "align_threshold": align_t,
                        "claim_gen_key": claim_gen_key,
                        "dedup_threshold": dedup_t,
                        "dedup_strategy": dedup_strat,
                        "dataset_name": dataset_name,
                        "records": results_for_this_config
                    }
                    results_all.append(config_result)

        # end for dedup_t
    # end for claim_gen_key

    # Finally, optionally save or return
    # Return the entire results for this dataset & method
    return results_all


# ------------------------------------------------------------------------------
# 4.2 Utility to get “raw scores” from an aligner without thresholding
#     The simplest approach is to replicate each aligner’s scoring logic.
# ------------------------------------------------------------------------------
def get_raw_alignment_scores(aligner, system_claims: List[str], reference_acus: List[str], method: str):
    """
    Returns a 2D list of shape [len(system_claims)][len(reference_acus)] with
    the raw alignment “score”:
      - Cosine sim for embeddings
      - Probability of entailment for NLI
      - ROUGE-1 F for ROUGE
    """
    # We can simply temporarily set aligner.threshold = -999, call `align()`, then extract the
    # computed scores if each aligner has a small hack to store them. But that requires modifying the aligner code.
    #
    # Alternatively, replicate the scoring steps. For clarity, let's replicate:

    import numpy as np
    from rouge_score import rouge_scorer

    num_sys = len(system_claims)
    num_ref = len(reference_acus)

    if isinstance(aligner, EmbeddingAligner):
        # 1) encode embeddings
        sys_embs, ref_embs = aligner._encode_items(system_claims, reference_acus)
        # 2) compute all pairwise cosines
        scores = [[0.0] * num_ref for _ in range(num_sys)]
        for i in range(num_sys):
            for j in range(num_ref):
                # same logic as _compute_score
                sim = float(
                    np.dot(sys_embs[i], ref_embs[j]) /
                    (np.linalg.norm(sys_embs[i]) * np.linalg.norm(ref_embs[j]) + 1e-9)
                )
                scores[i][j] = sim
        return scores

    elif isinstance(aligner, EntailmentAligner):
        # We'll do the same approach. We can do it in a more efficient “batch” manner if you prefer,
        # but here is a straightforward approach:
        scores = [[0.0] * num_ref for _ in range(num_sys)]
        for i in range(num_sys):
            for j in range(num_ref):
                # check cache first
                cached = aligner.cache.get_entailment_probability(system_claims[i], reference_acus[j])
                if cached is not None:
                    prob_entail = cached
                else:
                    prob_entail = aligner._infer_entailment(system_claims[i], reference_acus[j])
                    aligner.cache.set_entailment_probability(system_claims[i], reference_acus[j], prob_entail)

                scores[i][j] = prob_entail
        return scores

    else:
        # ROUGE
        # aligner = RougeAligner(threshold=0.0) has a self.scorer = ...
        # We'll compute rouge1_f. Because your original code does that:
        #    rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge_scorer_ = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = [[0.0] * num_ref for _ in range(num_sys)]
        for i, s_claim in enumerate(system_claims):
            for j, r_acu in enumerate(reference_acus):
                r = rouge_scorer_.score(target=r_acu, prediction=s_claim)
                scores[i][j] = r["rouge1"].fmeasure
        return scores


# ------------------------------------------------------------------------------
# 5. “Main” routine that runs everything
# ------------------------------------------------------------------------------
def main():
    print("Loading dataset(s)...")
    all_data = load_rose_dataset(USE_SMALL_TEST)

    # We store final results across all methods & datasets
    grand_results = []

    for method in METHODS:
        print(f"\n=== Method: {method} ===")
        for dataset_name in DATASET_NAMES:
            print(f"Processing dataset: {dataset_name}")
            dataset_results = run_experiments_for_dataset(dataset_name, method, all_data)
            grand_results.extend(dataset_results)

    # 5.1 Optionally save everything to one JSON
    out_path = Path("all_experiments_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(grand_results, f, indent=2)
    print(f"\n[Done] Wrote all experiment results to {out_path}")

    # 5.2 (Optional) If we used an EmbeddingAligner or NLI aligner, we can save cache:
    #   aligner.save_alignment_cache() is already invoked within run_experiments_for_dataset
    #   *if* you call it. (In the example above, we only do it at the very end, once.)


if __name__ == "__main__":
    timer = Timer()
    timer.start()
    main()
    timer.stop()
    timer.print_elapsed_time()
