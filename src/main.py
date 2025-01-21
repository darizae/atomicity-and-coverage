import subprocess

SMALL_TEST = False
CLAIM_GEN_MODEL = "gpt-3.5-turbo"


def main():
    # Define the experiments you want to run
    # Only these four parameters vary: method, threshold, claim_gen_key, small_test

    experiment_configs = [
        {
            "method": "rouge",
            "threshold": 0.3,
            "claim_gen_key": CLAIM_GEN_MODEL,
        },
        {
            "method": "embedding",
            "threshold": 0.7,
            "claim_gen_key": CLAIM_GEN_MODEL,
        },
        {
            "method": "embedding",
            "threshold": 0.8,
            "claim_gen_key": CLAIM_GEN_MODEL,
        },
        {
            "method": "entailment",
            "threshold": 0.9,
            "claim_gen_key": "distilled_t5",
        },
        # Add more experiments if you like
    ]

    for i, exp_conf in enumerate(experiment_configs, start=1):
        print(f"\n[Experiment {i}] Params: {exp_conf}")

        cmd = [
            "python", "metrics/run_alignment.py",
            "--method", exp_conf["method"],
            "--threshold", str(exp_conf["threshold"]),
            "--claim_gen_key", exp_conf["claim_gen_key"]
        ]
        # If small_test is True, we add the flag (no value needed)
        if SMALL_TEST:
            cmd.append("--small_test")

        subprocess.run(cmd, check=True)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
