import logging
from typing import List, Dict
from collections import defaultdict
from rouge_score import rouge_scorer
import numpy as np

logger = logging.getLogger(__name__)


def match_claims(
        system_claims: List[str],
        reference_acus: List[str],
        method: str = "rouge",
        threshold: float = 0.3,
        **kwargs
) -> Dict[int, List[int]]:
    """
    General interface to match system claims with reference ACUs.

    :param system_claims: List of system-generated claims.
    :param reference_acus: List of reference ACUs.
    :param method: One of "rouge", "embedding", or "entailment".
    :param threshold: Matching threshold. Interpretation varies by method.
    :param kwargs: Additional method-specific arguments (e.g. model, device, caches).
    :return: alignment_map = { sys_idx: [ref_idx, ...], ... }
    """
    alignment_map = {i: [] for i in range(len(system_claims))}

    if method == "rouge":
        alignment_map = _match_rouge(system_claims, reference_acus, threshold, **kwargs)
    elif method == "embedding":
        alignment_map = _match_embedding(system_claims, reference_acus, threshold, **kwargs)
    elif method == "entailment":
        alignment_map = _match_entailment(system_claims, reference_acus, threshold, **kwargs)
    else:
        logger.warning(f"Unknown method '{method}'. Falling back to 'rouge'.")
        alignment_map = _match_rouge(system_claims, reference_acus, threshold, **kwargs)

    return alignment_map


# -------------------------------------------------------------------
# 3.1 ROUGE-BASED MATCHING
# -------------------------------------------------------------------

def _match_rouge(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float,
        **kwargs
) -> Dict[int, List[int]]:
    """
    Uses the python `rouge-score` library if available. Otherwise, uses naive token overlap.
    threshold: the minimum F-measure of ROUGE-1 or token overlap ratio to count as a match.
    """
    alignment_map = defaultdict(list)

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    for i, s_claim in enumerate(system_claims):
        for j, r_acu in enumerate(reference_acus):
            scores = scorer.score(r_acu, s_claim)  # (reference, hypothesis) or vice versa
            rouge1_f = scores["rouge1"].fmeasure
            if rouge1_f >= threshold:
                alignment_map[i].append(j)

    return dict(alignment_map)


# -------------------------------------------------------------------
# 3.2 EMBEDDING-BASED MATCHING (with optional caching)
# -------------------------------------------------------------------

# A global dictionary for caching embeddings (text -> np.array)
# Alternatively, keep it in a class or a more sophisticated cache.
EMBED_CACHE = {}


def _match_embedding(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float,
        model=None,
        device: str = "cpu",
        **kwargs
) -> Dict[int, List[int]]:
    """
    Embedding-based matching using a model that provides sentence embeddings.
    :param threshold: Minimum cosine similarity to consider a match.
    :param model: A sentence-transformers or similar model with `.encode()`.
    :param device: "cpu" or "cuda"
    """
    alignment_map = defaultdict(list)

    if model is None:
        logger.warning("Embedding model not provided. Returning empty alignment.")
        return dict(alignment_map)

    # ---- (1) Batch encode system_claims and reference_acus (faster) ----
    sys_embeddings = _batch_get_embeddings(system_claims, model, device)
    ref_embeddings = _batch_get_embeddings(reference_acus, model, device)

    # ---- (2) For each sys-claim, compare to each ref-claim
    for i, s_emb in enumerate(sys_embeddings):
        for j, r_emb in enumerate(ref_embeddings):
            sim = _cosine_similarity(s_emb, r_emb)
            if sim >= threshold:
                alignment_map[i].append(j)

    return dict(alignment_map)


def _batch_get_embeddings(texts: List[str], model, device: str):
    """
    Encodes a list of texts, caching to avoid repeated computation.
    """
    # We'll collect the embeddings for each text in the same order.
    embeddings = []
    # Identify which texts are not yet in the cache
    texts_to_encode = []
    indices_to_encode = []

    for idx, txt in enumerate(texts):
        if txt in EMBED_CACHE:
            embeddings.append(EMBED_CACHE[txt])
        else:
            embeddings.append(None)
            texts_to_encode.append(txt)
            indices_to_encode.append(idx)

    if texts_to_encode:
        # Perform batch encoding for those missing
        # e.g. for sentence-transformers:
        batch_embs = model.encode(texts_to_encode, device=device, show_progress_bar=False)
        for k, emb in enumerate(batch_embs):
            # Store in cache
            EMBED_CACHE[texts_to_encode[k]] = emb
            # Update in the main embeddings list
            actual_index = indices_to_encode[k]
            embeddings[actual_index] = emb

    return embeddings


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / ((np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-9))


# -------------------------------------------------------------------
# 3.3 NLI-BASED MATCHING
# -------------------------------------------------------------------

def _match_entailment(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float,
        nli_model=None,
        device: str = "cpu",
        **kwargs
) -> Dict[int, List[int]]:
    """
    NLI-based matching:
    For each system claim, check if it's entailed by or entails the reference ACU
    with probability >= threshold.

    Implementation detail depends on your NLI model interface.
    """
    alignment_map = defaultdict(list)
    if nli_model is None:
        logger.warning("NLI model not provided. Returning empty alignment.")
        return dict(alignment_map)

    # For example, if we want to see if the system_claim => reference_acu
    # is 'entailed' with probability >= threshold:
    for i, s_claim in enumerate(system_claims):
        for j, r_acu in enumerate(reference_acus):
            prob_entailment = _compute_entailment_probability(
                premise=s_claim,
                hypothesis=r_acu,
                model=nli_model,
                device=device
            )
            if prob_entailment >= threshold:
                alignment_map[i].append(j)

    return dict(alignment_map)


def _compute_entailment_probability(premise, hypothesis, model, device: str = "cpu") -> float:
    """
    Example of how you'd run an NLI model to get the 'entailment' probability.
    E.g., with HuggingFace Transformers:
       - You tokenize premise + hypothesis
       - Run forward pass
       - Extract the entailment probability from the logits
    """
    # Pseudocode example:
    """
    inputs = tokenizer(premise, hypothesis, return_tensors='pt').to(device)
    logits = model(**inputs).logits
    # Let's say the label indices are [ENTAILMENT=2, NEUTRAL=1, CONTRADICTION=0], e.g., MNLI
    probabilities = torch.softmax(logits, dim=-1)
    entailment_prob = probabilities[0][2].item()
    return entailment_prob
    """
    return 0.5  # Stub: replace with real code
