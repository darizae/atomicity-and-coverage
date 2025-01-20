from dataclasses import dataclass, field

from src.config.models_config import EMBEDDING_MODELS, ENTAILMENT_MODELS


@dataclass(frozen=True)
class AlignmentMethods:
    ROUGE: str = "rouge"
    EMBEDDING: str = "embedding"
    ENTAILMENT: str = "entailment"


@dataclass
class EmbeddingModelConfig:
    model_name: str = EMBEDDING_MODELS["miniLM"]["model_name"]
    threshold: float = 0.7


@dataclass
class EntailmentModelConfig:
    model_name: str = ENTAILMENT_MODELS["roberta"]["model_name"]
    threshold: float = 0.9


@dataclass
class AlignmentConfig:
    method: str = AlignmentMethods.ROUGE
    threshold: float = 0.3
    device: str = "cpu"
    embedding_config: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    entailment_config: EntailmentModelConfig = field(default_factory=EntailmentModelConfig)
    cache_path: str = None
    claim_gen_key: str = "distilled_t5"
    dataset_name: str = None
