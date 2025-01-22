from dataclasses import dataclass
from typing import Type, Optional

from src.claims.claim_generator import BaseClaimGenerator, HuggingFaceSeq2SeqGenerator, HuggingFaceCausalGenerator, \
    OpenAIClaimGenerator, JanLocalClaimGenerator


SG_PREFIX = "system_claims_"


@dataclass
class ClaimGenerationModelConfig:
    name: str
    generator_cls: Type[BaseClaimGenerator]
    type: str  # e.g. "seq2seq", "causal", "openai", "local"
    tokenizer_class: Optional[str] = None
    model_class: Optional[str] = None
    endpoint_url: Optional[str] = None
    claims_field: Optional[str] = None


def get_claim_generation_model_config(model_key: str) -> ClaimGenerationModelConfig:
    if model_key == "distilled_t5":
        config = ClaimGenerationModelConfig(
            name="Babelscape/t5-base-summarization-claim-extractor",
            generator_cls=HuggingFaceSeq2SeqGenerator,
            type="seq2seq",
            tokenizer_class="transformers.T5Tokenizer",
            model_class="transformers.T5ForConditionalGeneration"
        )
    elif model_key == "llama2_7b":
        config = ClaimGenerationModelConfig(
            name="/works/data0/danielarizae/models/llama-2-7b-chat-hf",
            generator_cls=HuggingFaceCausalGenerator,
            type="local",
            tokenizer_class="transformers.LlamaTokenizer",
            model_class="transformers.LlamaForCausalLM"
        )
    elif model_key == "gpt-3.5-turbo":
        config = ClaimGenerationModelConfig(
            name="gpt-3.5-turbo",
            generator_cls=OpenAIClaimGenerator,
            type="openai"
        )
    elif model_key == "llama3_1b":
        config = ClaimGenerationModelConfig(
            name="llama3.2-1b-instruct",
            generator_cls=JanLocalClaimGenerator,
            type="local",
            endpoint_url="http://127.0.0.1:1337/v1/chat/completions",
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    config.claims_field = f"{SG_PREFIX}{model_key}"
    return config
