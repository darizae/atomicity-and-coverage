from src.config.paths import AlignmentPaths

ALIGNMENT_PATHS = AlignmentPaths()

CLAIM_GENERATION_MODELS = {
    "distilled_t5": {
        "name": "Babelscape/t5-base-summarization-claim-extractor",
        "claims_field": "system_claims_t5",
        "tokenizer_class": "transformers.T5Tokenizer",
        "model_class": "transformers.T5ForConditionalGeneration",
        "type": "seq2seq"  # This is helpful to identify that it's a seq2seq pipeline
    },
    "openai_gpt35": {
        "name": "gpt-3.5-turbo",  # OpenAI model name
        "claims_field": "system_claims_gpt35",
        "type": "openai",  # Let's define a new type for “OpenAI-based calls”
    },
    "llama2": {
        "name": "meta-llama/Llama-2-7b-chat-hf",  # an example Hugging Face LLaMA
        "claims_field": "system_claims_llama",
        "tokenizer_class": "transformers.LlamaTokenizer",
        "model_class": "transformers.LlamaForCausalLM",
        "type": "causal",  # Let’s define another type for “causal LM in huggingface”
    },
    # etc.
}


EMBEDDING_MODELS = {
    "miniLM": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_file": ALIGNMENT_PATHS.miniLM_cache_file,
        "threshold": 0.7
    },
    "mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "cache_file": ALIGNMENT_PATHS.mpnet_cache_file,
        "threshold": 0.65
    },
    # Add more if needed...
}

ENTAILMENT_MODELS = {
    "roberta": {
        "model_name": "roberta-large-mnli",
        "cache_file": ALIGNMENT_PATHS.roberta_mnli_cache_file,
        "threshold": 0.9
    },
    "bart": {
        "model_name": "facebook/bart-large-mnli",
        "cache_file": ALIGNMENT_PATHS.bart_mnli_cache_file,
        "threshold": 0.9
    }
}

