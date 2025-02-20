from .bipartite_entailment_alignment import BipartiteEntailmentAligner
from .config import AlignmentConfig, AlignmentMethods
from .rouge_alignment import RougeAligner
from .embedding_alignment import EmbeddingAligner
from .entailment_alignment import EntailmentAligner


def create_aligner(config: AlignmentConfig):
    """
    Factory to instantiate the correct alignment object based on the AlignmentConfig.

    :param config: An instance of AlignmentConfig.
    :return: An instance of BaseAligner.
    """
    match config.method.lower():
        case AlignmentMethods.ROUGE:
            return RougeAligner(threshold=config.threshold)

        case AlignmentMethods.EMBEDDING:
            if not config.embedding_config.model_name:
                raise ValueError("EmbeddingAligner requires a valid embedding model name.")
            return EmbeddingAligner(
                model_name=config.embedding_config.model_name,
                threshold=config.embedding_config.threshold,
                device=config.device,
                cache_path=config.cache_path
            )

        case AlignmentMethods.ENTAILMENT:
            if not config.entailment_config.model_name:
                raise ValueError("EntailmentAligner requires a valid NLI model name.")
            return EntailmentAligner(
                model_name=config.entailment_config.model_name,
                threshold=config.entailment_config.threshold,
                device=config.device,
                cache_path=config.cache_path
            )

        case AlignmentMethods.ENTAILMENT_BIPARTITE:
            if not config.entailment_config.model_name:
                raise ValueError("BipartiteEntailmentAligner requires an NLI model name.")
            return BipartiteEntailmentAligner(
                model_name=config.entailment_config.model_name,
                # The threshold might be a "floor" in bipartite approach
                threshold=config.entailment_config.threshold,
                device=config.device,
                cache_path=config.cache_path
            )

        case _:
            raise ValueError(f"Unknown alignment method: {config.method}")
