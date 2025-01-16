#!/bin/bash

# SLURM directives
#SBATCH --job-name=generate_claims
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00         # Adjust based on dataset size
#SBATCH --partition=gpu         # Or specify a CPU partition if GPUs aren't needed
#SBATCH --gres=gpu:1            # Number of GPUs required
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G               # Adjust memory as needed

# --- Enhanced Logging Start ---
echo "Job started on $(hostname) at $(date)"
echo "GPU status at start:"
nvidia-smi
echo "Free memory:"
free -h
echo "Disk usage (current directory):"
df -h .
# --- Enhanced Logging End ---

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate atomicity

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd):$(pwd)/src
export PYTHONUNBUFFERED=1

# Define the datasets and model
datasets=("cnndm_test")
model_key="distilled_t5"

# Iterate over the datasets
for dataset in "${datasets[@]}"; do
    echo "[$(date)] Processing dataset: $dataset on $(hostname)"

    # Optionally log GPU usage before each run
    echo "GPU status before processing $dataset:"
    nvidia-smi

    python src/claims/generate_claims.py --dataset_name "$dataset" --model_key "$model_key" --batch_size 32 --max_length 512

    if [ $? -ne 0 ]; then
        echo "Error encountered while processing $dataset at $(date). Check logs for details."
        exit 1
    fi

    echo "[$(date)] Finished processing $dataset"
done

echo "All datasets processed successfully!"

# --- Enhanced Logging End of Job ---
echo "Job finished at $(date)"
echo "GPU status at end:"
nvidia-smi
free -h
df -h .
# --- End Logging ---
