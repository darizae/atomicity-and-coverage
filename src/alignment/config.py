from dataclasses import dataclass
from typing import Optional

from src.utils.paths import AlignmentPaths

ALIGNMENT_PATHS = AlignmentPaths()


@dataclass
class EntailmentModelDefinition:
    key: str
    model_name: str
    cache_file: str
    threshold: float


@dataclass
class EmbeddingModelDefinition:
    key: str
    model_name: str
    cache_file: str
    threshold: float


def get_entailment_model_definition(model_key: str) -> EntailmentModelDefinition:
    """
    Return a typed EntailmentModelDefinition for the given model_key.
    """
    paths = ALIGNMENT_PATHS

    if model_key == "roberta":
        return EntailmentModelDefinition(
            key=model_key,
            model_name="roberta-large-mnli",
            cache_file=paths.roberta_mnli_cache_file,
            threshold=0.9
        )
    elif model_key == "bart":
        return EntailmentModelDefinition(
            key=model_key,
            model_name="facebook/bart-large-mnli",
            cache_file=paths.bart_mnli_cache_file,
            threshold=0.9
        )
    else:
        raise ValueError(f"Unknown entailment model key: '{model_key}'")


def get_embedding_model_definition(model_key: str) -> EmbeddingModelDefinition:
    """
    Return a typed EmbeddingModelDefinition for the given model_key.
    """
    paths = AlignmentPaths()

    if model_key == "miniLM":
        return EmbeddingModelDefinition(
            key=model_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_file=paths.miniLM_cache_file,
            threshold=0.7
        )
    elif model_key == "mpnet":
        return EmbeddingModelDefinition(
            key=model_key,
            model_name="sentence-transformers/all-mpnet-base-v2",
            cache_file=paths.mpnet_cache_file,
            threshold=0.65
        )
    else:
        raise ValueError(f"Unknown embedding model key: '{model_key}'")


@dataclass(frozen=True)
class AlignmentMethods:
    ROUGE: str = "rouge"
    EMBEDDING: str = "embedding"
    ENTAILMENT: str = "entailment"
    ENTAILMENT_BIPARTITE: str = "entailment_bipartite"
    EMBEDDING_BIPARTITE: str = "embedding_bipartite"


@dataclass
class EmbeddingModelConfig:
    model_name: str
    cache_file: str
    threshold: float


@dataclass
class EntailmentModelConfig:
    model_name: str
    cache_file: str
    threshold: Optional[float] = None


@dataclass
class AlignmentConfig:
    method: str = AlignmentMethods.ROUGE
    threshold: float = 0.3
    device: str = "cpu"
    embedding_config: Optional[EmbeddingModelConfig] = None
    entailment_config: Optional[EntailmentModelConfig] = None
    cache_path: Optional[str] = None
    claim_gen_key: str = "distilled_t5"
    reference_claims_key: str = "reference_acus_deduped_0.9_select_longest"
    dataset_name: Optional[str] = None

