from .rouge_alignment import RougeAligner
from .embedding_alignment import EmbeddingAligner
from .entailment_alignment import EntailmentAligner


def create_aligner(
        method: str,
        threshold: float = 0.3,
        model=None,
        tokenizer=None,
        device: str = "cpu"
):
    """
    Factory to instantiate the correct alignment object
    based on method string.

    :param method: "rouge", "embedding", or "entailment"
    :param threshold: matching threshold
    :param model: embedding or NLI model, if required
    :param tokenizer: for NLI, if required
    :param device: "cpu" or "cuda"
    :return: an instance of BaseAligner
    """
    method = method.lower()

    if method == "rouge":
        return RougeAligner(threshold=threshold)
    elif method == "embedding":
        if model is None:
            raise ValueError("EmbeddingAligner requires an embedding model.")
        return EmbeddingAligner(model=model, threshold=threshold, device=device)
    elif method == "entailment":
        if model is None or tokenizer is None:
            raise ValueError("EntailmentAligner requires an NLI model and tokenizer.")
        return EntailmentAligner(nli_model=model, tokenizer=tokenizer, threshold=threshold, device=device)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
