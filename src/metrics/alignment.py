import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def match_claims(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float = 0.5,
        method: str = "rouge",
        **kwargs
) -> Dict[str, List[int]]:
    """
    Main interface function that aligns system claims with reference ACUs.
    Returns a dictionary where keys are system claim indices and
    values are lists of matched reference ACU indices.

    :param system_claims: List of system-generated claims (S).
    :param reference_acus: List of human-annotated claims (G).
    :param threshold: A float that determines the minimum similarity
                      or entailment probability to consider a match.
    :param method: The matching technique to use ('rouge', 'embedding', 'entailment').
    :param kwargs: Additional arguments for each method (e.g., model, device).
    :return: Dict: { system_claim_index: [matched_ref_acu_indices], ... }
    """

    # A dictionary to hold alignment information
    alignment_map = {i: [] for i in range(len(system_claims))}

    # Choose matching method
    if method == "rouge":
        alignment_map = _rouge_based_matching(system_claims, reference_acus, threshold)
    elif method == "embedding":
        # e.g. use sentence-transformers or similar
        alignment_map = _embedding_based_matching(system_claims, reference_acus, threshold, **kwargs)
    elif method == "entailment":
        # e.g. use an NLI model
        alignment_map = _entailment_based_matching(system_claims, reference_acus, threshold, **kwargs)
    else:
        logger.warning(f"Unrecognized method: {method}. Defaulting to 'rouge'.")
        alignment_map = _rouge_based_matching(system_claims, reference_acus, threshold)

    return alignment_map


def _rouge_based_matching(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float
) -> Dict[int, List[int]]:
    """
    Simple example using ROUGE or a token-overlap measure as a proxy for
    semantic similarity.
    """
    from collections import defaultdict
    alignment_map = defaultdict(list)

    # Pseudocode or actual code using a library like `rouge_score`
    # For demonstration, we’ll do a naive token overlap ratio.
    for i, s_claim in enumerate(system_claims):
        s_tokens = set(s_claim.lower().split())
        for j, r_acu in enumerate(reference_acus):
            r_tokens = set(r_acu.lower().split())
            # naive "overlap" ratio
            overlap = len(s_tokens.intersection(r_tokens)) / max(len(s_tokens), 1)
            if overlap >= threshold:
                alignment_map[i].append(j)
    return dict(alignment_map)


def _embedding_based_matching(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float,
        model: Any = None,
        device: str = "cpu"
) -> Dict[int, List[int]]:
    """
    Example placeholder for embedding-based similarity approach.
    """
    from collections import defaultdict
    alignment_map = defaultdict(list)

    # Suppose 'model' is a sentence-transformers model
    # get embeddings
    s_embeddings = model.encode(system_claims, device=device)
    r_embeddings = model.encode(reference_acus, device=device)

    # compute pairwise similarities
    # This can be a simple cosine similarity for each pair
    for i, s_emb in enumerate(s_embeddings):
        for j, r_emb in enumerate(r_embeddings):
            similarity = _cosine_similarity(s_emb, r_emb)
            if similarity >= threshold:
                alignment_map[i].append(j)

    return dict(alignment_map)


def _entailment_based_matching(
        system_claims: List[str],
        reference_acus: List[str],
        threshold: float,
        nli_model: Any = None,
        device: str = "cpu"
) -> Dict[int, List[int]]:
    """
    Example placeholder for an NLI-based approach that checks if
    system_claim => reference_ACU is entailed with probability >= threshold.
    """
    from collections import defaultdict
    alignment_map = defaultdict(list)

    # For each system claim, run NLI against each reference ACU
    for i, s_claim in enumerate(system_claims):
        for j, r_acu in enumerate(reference_acus):
            # get entailment probability
            prob_entailment = _compute_entailment_probability(nli_model, s_claim, r_acu, device=device)
            if prob_entailment >= threshold:
                alignment_map[i].append(j)

    return dict(alignment_map)


def _cosine_similarity(vec1, vec2):
    import numpy as np
    return float(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)
    )


def _compute_entailment_probability(nli_model, premise: str, hypothesis: str, device: str = "cpu") -> float:
    """
    Stub function: run an NLI model forward pass,
    returning the probability that `premise => hypothesis`.
    """
    # In practice, you'd do something like:
    # inputs = nli_model.tokenize(premise, hypothesis, return_tensors="pt").to(device)
    # outputs = nli_model(**inputs)
    # prob_entail = softmax(outputs.logits, dim=-1)[ENTAILMENT_INDEX]
    # return float(prob_entail)
    return 0.75  # placeholder