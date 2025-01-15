from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentConfig:
    method: str = "rouge"  # Default alignment method
    threshold: float = 0.3  # Default matching threshold
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Default embedding model
    entailment_model: str = "facebook/bart-large-mnli"  # Default NLI model
    device: str = "cpu"  # Device for running models
