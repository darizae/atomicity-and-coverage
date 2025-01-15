#!/bin/bash

# SLURM directives
#SBATCH --job-name=generate_claims
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00 # Adjust based on dataset size
#SBATCH --partition=gpu # Or specify a CPU partition if GPUs aren't needed
#SBATCH --gres=gpu:1    # Number of GPUs required
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G       # Adjust memory as needed

# Activate the environment
source ~/envs/atomicity/bin/activate

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
