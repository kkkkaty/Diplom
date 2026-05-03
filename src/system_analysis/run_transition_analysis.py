#файл запуска для поиска смены нидингов

import os
import yaml
import numpy as np

from src.system_analysis.transition_tools_v2 import (
    find_interesting_transitions_fast,
    refine_interesting_transitions,
    copy_best_case_figures,
)


config_path = r"C:/Lobach4/Diplom/kneadings-master/config/kneadings_pendulums.yaml"
npy_path = r"C:/Lobach4/tu/kneadings-master1/output/kneadings_pendulums1.npy"

base_output_dir = os.path.dirname(npy_path)

fast_dir = os.path.join(base_output_dir, "interesting_transition_scan_fast")
refined_dir = os.path.join(base_output_dir, "interesting_transition_scan_refined")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

kneading_map = np.load(npy_path)


fast_hits = find_interesting_transitions_fast(
    config=config,
    kneading_map_flat=kneading_map,
    output_dir=fast_dir,
    row_step=1,
    max_results=3000,
    n_steps_traj=3000,
    stride_traj=1,
)

print("\nFAST TOP transitions:\n")

for i, item in enumerate(fast_hits):
    print(
        i,
        "row =", item["row_index"],
        "transition =", item["transition_number"],
        "score =", item["score"],
        "common_score =", item.get("common_score"),
        "has_common_eq =", item.get("has_common_eq"),
    )


refined_hits = refine_interesting_transitions(
    config=config,
    kneading_map_flat=kneading_map,
    coarse_hits=fast_hits,
    output_dir=refined_dir,
    max_results=3,
    n_steps_traj=30000,
    stride_traj=1,
)

print("\nREFINED TOP transitions:\n")

for i, item in enumerate(refined_hits):
    print(
        i,
        "row =", item["row_index"],
        "transition =", item["transition_number"],
        "score =", item["score"],
        "common_score =", item.get("common_score"),
        "has_common_eq =", item.get("has_common_eq"),
        "dir =", item["result"].output_dir,
    )


if refined_hits:
    best_result = refined_hits[0]["result"]

    best_dir = os.path.join(refined_dir, "BEST_READY_FIGURES")
    copy_best_case_figures(best_result, best_dir)

    print("\nЛучшие готовые графики скопированы в:", best_dir)


print("\nDone")