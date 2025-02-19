import subprocess

SMALL_TEST = False
CLAIM_GEN_MODELS = [
    "gpt_maximize_atomicity",
    "gpt_maximize_coverage",
    "gpt_granularity_low",
    "gpt_granularity_high",
    "kg_based_claims"
]


def main():
    # Define the experiments to run
    experiment_configs = []

    methods = ["entailment"]
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.7]

    for model in CLAIM_GEN_MODELS:
        for method in methods:
            for threshold in thresholds:
                experiment_configs.append({
                    "method": method,
                    "threshold": threshold,
                    "claim_gen_key": model,
                })

    for i, exp_conf in enumerate(experiment_configs, start=1):
        print(f"\n[Experiment {i}] Params: {exp_conf}")

        cmd = [
            "python", "metrics/run_alignment.py",
            "--method", exp_conf["method"],
            "--threshold", str(exp_conf["threshold"]),
            "--claim_gen_key", exp_conf["claim_gen_key"]
        ]

        if SMALL_TEST:
            cmd.append("--small_test")

        subprocess.run(cmd, check=True)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
