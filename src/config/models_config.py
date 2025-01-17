MODELS = {
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
        "cache_file": "embedding_cache_all_MiniLM.pkl",
        "threshold": 0.7
    },
    "mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "cache_file": "embedding_cache_mpnet.pkl",
        "threshold": 0.65
    },
    # Add more if needed...
}

