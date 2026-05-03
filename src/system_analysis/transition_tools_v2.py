from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.system_analysis.analyze_separatrix_transition_pendulums_v2 import (
    TransitionAnalysisResult,
    analyze_separatrix_transition_v2,
    build_parameter_grid_from_config,
    extract_horizontal_scan,
    find_code_transitions_on_scan,
    reshape_map,
)


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ МЕТРИКИ
# =========================================================

def _final_state_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(a[n - 1] - b[n - 1]))


def _trajectory_gap_score(before_traj: np.ndarray, after_traj: np.ndarray) -> float:
    n = min(len(before_traj), len(after_traj))
    if n == 0:
        return 0.0
    start = max(0, (3 * n) // 4)
    gap = np.linalg.norm(before_traj[start:n] - after_traj[start:n], axis=1)
    return float(np.mean(gap)) if gap.size else 0.0


# =========================================================
# СКОРИНГ
# =========================================================

def score_transition_result(result: TransitionAnalysisResult) -> Dict[str, Any]:
    probe_map = {p.label: p for p in result.probes}
    before = probe_map["before"]
    after = probe_map["after"]

    divergence_final = _final_state_distance(before.trajectory, after.trajectory)
    divergence_tail = _trajectory_gap_score(before.trajectory, after.trajectory)

    if result.common_eq is None or result.common_eq_meta is None:
        common_score = float("inf")
        is_good_separator = False
        mins = []
    else:
        common_score = float(result.common_eq_meta["score"])
        is_good_separator = common_score < -1.0
        mins = result.common_eq_meta.get("mins", [])

    if is_good_separator:
        score = (
            -10.0 * common_score
            + 0.2 * np.log10(divergence_final + 1e-12)
            + 0.2 * np.log10(divergence_tail + 1e-12)
        )
    else:
        score = -1e9

    return {
        "row_index": int(result.row_index),
        "transition_number": (
            int(result.transition_left_index),
            int(result.transition_right_index),
        ),
        "score": float(score),
        "common_score": float(common_score),
        "mins": mins,
        "has_common_eq": bool(result.common_eq is not None),
        "result": result,
    }


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПЕРЕХОДОВ
# =========================================================

def _resolve_transition_index(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    row_index: int,
    target_pair: Tuple[int, int],
) -> int:
    scan = extract_horizontal_scan(kneading_map_flat, config, row_index=row_index)
    transitions = find_code_transitions_on_scan(scan)

    if target_pair not in transitions:
        raise ValueError(
            f"Transition pair {target_pair} not found at row {row_index}. "
            f"Available: {transitions[:10]}{'...' if len(transitions) > 10 else ''}"
        )

    return transitions.index(target_pair)


# =========================================================
# ЭТАП 1 — БЫСТРЫЙ СКРИНИНГ
# =========================================================

def find_interesting_transitions_fast(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    output_dir: str,
    row_step: int = 50,
    max_results: int = 20,
    closeness: float = 0.01,
    n_steps_traj: int = 3000,
    stride_traj: int = 1,
    prefer_symmetric_equilibria: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    _, _, _, rows = build_parameter_grid_from_config(config)

    ranked: List[Dict[str, Any]] = []

    for row_index in range(0, rows, row_step):
        print(f"[FAST] row {row_index}")

        scan = extract_horizontal_scan(kneading_map_flat, config, row_index=row_index)
        transitions = find_code_transitions_on_scan(scan)

        for tr_no in range(len(transitions)):
            try:
                case_dir = os.path.join(output_dir, f"row_{row_index:04d}_tr_{tr_no:03d}")

                result = analyze_separatrix_transition_v2(
                    config=config,
                    kneading_map_flat=kneading_map_flat,
                    output_dir=case_dir,
                    row_index=row_index,
                    transition_number=tr_no,
                    closeness=closeness,
                    dt_traj=None,
                    n_steps_traj=n_steps_traj,
                    stride_traj=stride_traj,
                    prefer_symmetric_equilibria=prefer_symmetric_equilibria,
                )

                ranked.append(score_transition_result(result))

            except Exception as e:
                print(f"  [FAST skip] row={row_index} tr={tr_no}: {e}")
                continue

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:max_results]


# =========================================================
# ЭТАП 2 — ТОЧНЫЙ АНАЛИЗ
# =========================================================

def refine_interesting_transitions(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    coarse_hits: List[Dict[str, Any]],
    output_dir: str,
    max_results: int = 3,
    closeness: float = 0.01,
    n_steps_traj: int = 30000,
    stride_traj: int = 1,
    prefer_symmetric_equilibria: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    refined: List[Dict[str, Any]] = []

    for i, item in enumerate(coarse_hits[:max_results]):
        row_index = int(item["row_index"])
        target_pair = tuple(item["transition_number"])

        print(f"[REFINE] case {i} row={row_index} transition_pair={target_pair}")

        try:
            tr_no = _resolve_transition_index(
                config=config,
                kneading_map_flat=kneading_map_flat,
                row_index=row_index,
                target_pair=target_pair,
            )

            result = analyze_separatrix_transition_v2(
                config=config,
                kneading_map_flat=kneading_map_flat,
                output_dir=os.path.join(output_dir, f"case_{i}"),
                row_index=row_index,
                transition_number=tr_no,
                closeness=closeness,
                dt_traj=None,
                n_steps_traj=n_steps_traj,
                stride_traj=stride_traj,
                prefer_symmetric_equilibria=prefer_symmetric_equilibria,
            )

            refined.append(score_transition_result(result))

        except Exception as e:
            print(f"  [REFINE skip] case={i}: {e}")
            continue

    refined.sort(key=lambda x: x["score"], reverse=True)
    return refined


# =========================================================
# КОПИРОВАНИЕ ГОТОВЫХ ГРАФИКОВ
# =========================================================

def copy_best_case_figures(result: TransitionAnalysisResult, target_dir: str) -> str:
    """
    Не строит графики заново.
    Просто копирует уже готовые картинки, которые построил analyze_separatrix_transition_v2.
    """

    os.makedirs(target_dir, exist_ok=True)

    figure_names = [
        "01_parameter_map_real_axes.png",
        "02_horizontal_scan.png",
        "03_phase_projections_4d.png",
        "05_distance_to_candidate_equilibria.png",
        "06_time_series_wrapped_unwrapped.png",
        "07_before_after_transition.png",
        "transition_report.json",
        "transition_summary.txt",
    ]

    for name in figure_names:
        src = os.path.join(result.output_dir, name)
        dst = os.path.join(target_dir, name)

        if os.path.exists(src):
            shutil.copy2(src, dst)

    return target_dir


# =========================================================
# СУПЕР-ГРАФИК
# ОСТАВЛЯЕМ, НО ЛУЧШЕ ПОКА НЕ ИСПОЛЬЗОВАТЬ
# =========================================================

def make_super_transition_figure(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    result: TransitionAnalysisResult,
    filename: Optional[str] = None,
) -> str:
    if filename is None:
        filename = os.path.join(result.output_dir, "08_super_transition_figure.png")

    params_x, params_y, cols, rows = build_parameter_grid_from_config(config)
    arr2d = reshape_map(kneading_map_flat, cols, rows)
    scan = extract_horizontal_scan(kneading_map_flat, config, row_index=result.row_index)

    x_scan = np.array([p.x for p in scan], dtype=float)
    raw_scan = np.array([p.raw_value for p in scan], dtype=float)

    x_min = float(np.min(params_x))
    x_max = float(np.max(params_x))
    y_min = float(np.min(params_y))
    y_max = float(np.max(params_y))

    probe_map = {p.label: p for p in result.probes}
    before = probe_map["before"]
    after = probe_map["after"]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])

    ax_map = fig.add_subplot(gs[:, 0:2])
    ax_map.imshow(
        arr2d,
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
    )
    ax_map.axhline(result.y_scan_value, linestyle="--", linewidth=1.5)
    ax_map.scatter(
        [result.x_boundary_estimate],
        [result.y_scan_value],
        s=110,
        marker="x",
        color="red",
        label="выбранный переход",
    )
    ax_map.set_title("Карта параметров и выбранный переход")
    ax_map.set_xlabel(config["grid"]["first"].get("caption", config["grid"]["first"]["name"]))
    ax_map.set_ylabel(config["grid"]["second"].get("caption", config["grid"]["second"]["name"]))
    ax_map.legend()

    ax_scan = fig.add_subplot(gs[0, 2:4])
    ax_scan.plot(
        x_scan,
        raw_scan,
        color="black",
        marker="o",
        markersize=2.2,
        linewidth=1.2,
        label="значение нидинга",
    )
    ax_scan.axvline(result.x_boundary_estimate, color="red", linestyle="--", linewidth=1.5, label="смена кода/цвета")
    ax_scan.axvline(before.x_param, color="tab:orange", linestyle=":", linewidth=1.2, label="точка до границы")
    ax_scan.axvline(after.x_param, color="tab:green", linestyle=":", linewidth=1.2, label="точка после границы")
    ax_scan.set_title("Горизонтальный срез карты нидингов")
    ax_scan.set_xlabel(config["grid"]["first"].get("caption", config["grid"]["first"]["name"]))
    ax_scan.set_ylabel("raw kneading value")
    ax_scan.grid(True, alpha=0.3)
    ax_scan.legend(fontsize=8)

    sub = gs[1, 2:4].subgridspec(1, 2)
    axes = [fig.add_subplot(sub[0, i]) for i in range(2)]

    plot_t_max = 50.0

    for ax, p, color, title in [
        (axes[0], before, "tab:orange", "до границы"),
        (axes[1], after, "tab:green", "после границы"),
    ]:
        mask = p.time <= plot_t_max
        if not np.any(mask):
            mask = np.ones_like(p.time, dtype=bool)

        traj_plot = p.trajectory[mask]

        ax.plot(
            traj_plot[:, 0],
            traj_plot[:, 2],
            color=color,
            linewidth=2.0,
            label=f"сепаратриса {title}",
        )

        ax.scatter(
            [traj_plot[0, 0]],
            [traj_plot[0, 2]],
            color=color,
            marker="D",
            s=70,
            edgecolor="black",
            linewidth=0.5,
            label="начало траектории",
            zorder=5,
        )

        ax.scatter(
            [traj_plot[-1, 0]],
            [traj_plot[-1, 2]],
            color=color,
            marker="x",
            s=90,
            linewidth=2.5,
            label="конец траектории",
            zorder=6,
        )

        ax.scatter(
            [p.source_eq[0]],
            [p.source_eq[2]],
            color=color,
            marker="o",
            s=75,
            edgecolor="black",
            linewidth=0.5,
            label="исходное равновесие",
            zorder=7,
        )

        ax.scatter(
            [p.closest_eq[0]],
            [p.closest_eq[2]],
            color="red",
            marker="*",
            s=150,
            edgecolor="black",
            linewidth=0.6,
            label="ближайшее равновесие",
            zorder=8,
        )

        ax.set_title(f"{title}, t ≤ {plot_t_max:g}")
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Сепаратриса до / после границы", fontsize=16)

    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.savefig(filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return filename


__all__ = [
    "score_transition_result",
    "find_interesting_transitions_fast",
    "refine_interesting_transitions",
    "copy_best_case_figures",
    "make_super_transition_figure",
]