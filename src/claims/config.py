from dataclasses import dataclass
from typing import Type, Optional

from src.claims.claim_generator import BaseClaimGenerator, HuggingFaceSeq2SeqGenerator, HuggingFaceCausalGenerator, \
    OpenAIClaimGenerator, JanLocalClaimGenerator


@dataclass
class ClaimGenerationModelConfig:
    name: str
    claims_field: str
    generator_cls: Type[BaseClaimGenerator]
    type: str  # e.g. "seq2seq", "causal", "openai", "local"
    tokenizer_class: Optional[str] = None
    model_class: Optional[str] = None
    endpoint_url: Optional[str] = None


def get_claim_generation_model_config(model_key: str) -> ClaimGenerationModelConfig:
    if model_key == "distilled_t5":
        return ClaimGenerationModelConfig(
            name="Babelscape/t5-base-summarization-claim-extractor",
            claims_field="system_claims_distilled_t5",
            generator_cls=HuggingFaceSeq2SeqGenerator,
            type="seq2seq",
            tokenizer_class="transformers.T5Tokenizer",
            model_class="transformers.T5ForConditionalGeneration"
        )
    elif model_key == "llama2_7b_local":
        return ClaimGenerationModelConfig(
            name="/works/data0/jiang/model/huggingface/llama-7b-hf",
            claims_field="system_claims_llama2_local",
            generator_cls=HuggingFaceCausalGenerator,
            type="local",
            tokenizer_class="transformers.LlamaTokenizer",
            model_class="transformers.LlamaForCausalLM"
        )
    elif model_key == "gpt-3.5-turbo":
        return ClaimGenerationModelConfig(
            name="gpt-3.5-turbo",
            claims_field="system_claims_gpt35",
            generator_cls=OpenAIClaimGenerator,
            type="openai"
        )
    elif model_key == "llama3_1b":
        return ClaimGenerationModelConfig(
            name="llama3.2-1b-instruct",
            claims_field="system_claims_llama3.2-1b-instruct",
            generator_cls=JanLocalClaimGenerator,
            type="local",
            endpoint_url="http://127.0.0.1:1337/v1/chat/completions",
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")
