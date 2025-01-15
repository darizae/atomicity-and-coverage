#!/bin/bash

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

# Define the datasets and model
datasets=("cnndm_test" "cnndm_validation" "xsum" "samsum")
model_key="distilled_t5"

# Iterate over the datasets
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python src/claims/generate_claims.py --dataset_name "$dataset" --model_key "$model_key" --batch_size 32 --max_length 512
    if [ $? -ne 0 ]; then
        echo "Error encountered while processing $dataset. Check logs for details."
        exit 1
    fi
done

echo "All datasets processed successfully!"