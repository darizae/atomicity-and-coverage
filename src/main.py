import subprocess


def main():
    experiment_configs = []

    methods = ["rouge", "embedding", "entailment"]

    align_thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Dedup thresholds used in the reference JSON
    ref_dedup_thresholds = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Possibly just one strategy "select_longest", or more
    ref_dedup_strategies = ["select_longest"]

    claim_gen_models = ["distilled_t5", "gpt-3.5-turbo", "llama2_7b"]

    for model in claim_gen_models:
        for method in methods:
            for align_t in align_thresh_list:
                for dedup_t in ref_dedup_thresholds:
                    for dedup_strat in ref_dedup_strategies:
                        experiment_configs.append({
                            "method": method,
                            "align_threshold": align_t,
                            "claim_gen_key": model,
                            "ref_dedup_threshold": dedup_t,
                            "ref_dedup_strategy": dedup_strat
                        })

    for i, exp_conf in enumerate(experiment_configs, start=1):
        print(f"\n[Experiment {i}] {exp_conf}")

        cmd = [
            "python",
            "metrics/run_alignment.py",
            "--method", exp_conf["method"],
            "--threshold", str(exp_conf["align_threshold"]),
            "--claim_gen_key", exp_conf["claim_gen_key"],
        ]

        # If dedup_threshold is not None, we add it
        if exp_conf["ref_dedup_threshold"] is not None:
            cmd.extend(["--dedup_threshold", str(exp_conf["ref_dedup_threshold"])])

        # Similarly for dedup_strategy
        if exp_conf["ref_dedup_strategy"] is not None:
            cmd.extend(["--dedup_strategy", exp_conf["ref_dedup_strategy"]])

        subprocess.run(cmd, check=True)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
