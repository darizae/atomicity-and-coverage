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