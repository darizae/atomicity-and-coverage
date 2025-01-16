from typing import Dict, List


def compute_coverage(
        alignment_map: Dict[int, List[int]],
        total_ref_acus: int
) -> float:
    """
    Coverage = (# of matched reference claims) / (total # of reference claims).

    :param alignment_map: A dict { system_claim_idx: [matched_ref_acu_idx, ...], ... }.
    :param total_ref_acus: The total number of reference ACUs (|G|).
    :return: A float representing coverage in [0, 1].
    """
    matched_ref_indices = set()
    for ref_list in alignment_map.values():
        for r_idx in ref_list:
            matched_ref_indices.add(r_idx)

    coverage = len(matched_ref_indices) / total_ref_acus if total_ref_acus > 0 else 0.0
    return coverage


def compute_atomicity(
        alignment_map: Dict[int, List[int]],
        total_sys_claims: int
) -> float:
    """
    Atomicity = avg(1 / # of reference matches) across all system claims.

    :param alignment_map: A dict { system_claim_idx: [matched_ref_acu_idx, ...], ... }.
    :param total_sys_claims: The total number of system claims (|S|).
    :return: A float representing atomicity in [0, 1].
    """
    if total_sys_claims == 0:
        return 0.0

    atomicity_values = []
    for s_idx, ref_list in alignment_map.items():
        num_matches = len(ref_list)
        if num_matches == 0:
            atomicity_values.append(0.0)
        else:
            atomicity_values.append(1.0 / num_matches)

    return sum(atomicity_values) / total_sys_claims
