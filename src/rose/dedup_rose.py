from claim_deduplicator.multi_threshold_deduplicate import multi_threshold_deduplicate
from claim_deduplicator.strategies import select_longest

from src.utils.paths import RosePathsSmall, RosePaths


def run_multi_threshold_dedup_on_rose(
        input_json,
        output_json,
        thresholds=None,
        representative_strategy=select_longest,
        measure_redundancy=False,
        cluster_analysis_dir="cluster_logs",
        device=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    if thresholds is None:
        raise ValueError("thresholds cannot be None")

    multi_threshold_deduplicate(
        input_json_path=input_json,
        output_json_path=output_json,
        thresholds=thresholds,
        representative_selector=representative_strategy,
        model_name=model_name,
        device=device,
        measure_redundancy_flag=measure_redundancy,
        cluster_analysis_dir=cluster_analysis_dir
    )


if __name__ == "__main__":
    # input_json = RosePathsSmall.dataset_path
    # output_json = RosePathsSmall.dataset_path

    input_json = RosePaths.dataset_path
    output_json = RosePaths.dataset_path

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    representative_strategy = select_longest
    measure_redundancy = True

    cluster_analysis_dir = "data/cluster_logs"

    run_multi_threshold_dedup_on_rose(
        input_json=input_json,
        output_json=output_json,
        thresholds=thresholds,
        representative_strategy=representative_strategy,
        measure_redundancy=measure_redundancy,
        cluster_analysis_dir=cluster_analysis_dir,
    )