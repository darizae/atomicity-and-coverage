from src.config.paths import AlignmentPaths

ALIGNMENT_PATHS = AlignmentPaths()

CLAIM_GENERATION_MODELS = {
    "distilled_t5": {
        "name": "Babelscape/t5-base-summarization-claim-extractor",
        "claims_field": "system_claims_t5",
        "tokenizer_class": "transformers.T5Tokenizer",
        "model_class": "transformers.T5ForConditionalGeneration",
    },
    # Add more models as needed
    # "another_model": {
    #     "name": "some/other-model",
    #     "claims_field": "system_claims_other",
    # },
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
    "roberta-large-mnli": {
        "model_name": "roberta-large-mnli",
        "cache_file": ALIGNMENT_PATHS.roberta_mnli_cache_file,
        "threshold": 0.9
    },
    "bart-large-mnli": {
        "model_name": "facebook/bart-large-mnli",
        "cache_file": ALIGNMENT_PATHS.bart_mnli_cache_file,
        "threshold": 0.9
    }
}

