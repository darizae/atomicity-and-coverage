from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlignmentMethods:
    ROUGE: str = "rouge"
    EMBEDDING: str = "embedding"
    ENTAILMENT: str = "entailment"


@dataclass
class EmbeddingModelConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.7


@dataclass
class EntailmentModelConfig:
    model_name: str = "roberta-large-mnli"
    threshold: float = 0.9


@dataclass
class AlignmentConfig:
    method: str = AlignmentMethods.ROUGE
    threshold: float = 0.3
    device: str = "cpu"
    embedding_config: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    entailment_config: EntailmentModelConfig = field(default_factory=EntailmentModelConfig)
    cache_path: str = None
