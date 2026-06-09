from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from src.mapping.plot_kneadings import set_random_color_map

import matplotlib.pyplot as plt
import numpy as np

from src.system_analysis.get_inits import (
    build_separatrix_init_for_point,
    equilibrium_type,
    find_equilibria_pendulum,
    rk4_step,
)

EPS_LOG = 1e-30


@dataclass
class ScanPoint:
    i: int
    j: int
    x: float
    y: float
    raw_value: float
    code: str


@dataclass
class EquilibriumCandidate:
    point: np.ndarray
    nU: int
    nS: int
    nC: int
    eigvals: np.ndarray
    is_symmetric: bool


@dataclass
class ProbeResult:
    label: str
    side: str
    x_param: float
    y_param: float
    params: Tuple[float, float, float]
    source_eq: np.ndarray
    init_point: np.ndarray
    unstable_dir: np.ndarray
    branch_id: int
    trajectory: np.ndarray
    time: np.ndarray
    closest_eq: np.ndarray
    closest_eq_meta: Dict[str, Any]
    log_distance_to_best_eq: np.ndarray
    best_eq_min_log10: float
    best_eq_argmin_t: float
    best_eq_argmin_index: int
    candidate_eqs: List[EquilibriumCandidate]


@dataclass
class TransitionAnalysisResult:
    row_index: int
    transition_left_index: int
    transition_right_index: int
    left_scan_point: ScanPoint
    right_scan_point: ScanPoint
    x_boundary_estimate: float
    y_scan_value: float
    probes: List[ProbeResult]
    candidate_equilibria: List[Dict[str, Any]]
    output_dir: str
    common_eq: Optional[np.ndarray] = None
    common_eq_meta: Optional[Dict[str, Any]] = None


def wrap_angle_0_2pi(phi: np.ndarray | float) -> np.ndarray | float:
    return np.mod(phi, 2.0 * np.pi)


def decode_base25_weighted(x: float, length: int) -> str:
    if x < 0:
        return str(x)  # для -0.1/-0.2/-0.3
    symbols: List[str] = []
    v = float(x)
    # С основанием 25 декодируем до 10 символов
    safe_length = min(length, 10)
    for _ in range(safe_length):
        v *= 25.0
        combined_symbol = int(v + 1e-11)
        # Ограничиваем значение символа от 0 до 44
        combined_symbol = max(0, min(24, combined_symbol))
        m1 = combined_symbol // 5
        m2 = combined_symbol % 5
        # Записываем сам индекс совместного события от 00 до 24
        symbols.append(f"{combined_symbol:02d}")
        # Оставляем только дробную часть для следующего шага
        v -= combined_symbol
    return " - ".join(symbols)


def _generate_parameters_2d(
    start_x, start_y, up_n, down_n, left_n, right_n,
    up_step, down_step, left_step, right_step
):
    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    total = cols * rows

    params_x = np.empty(total, dtype=np.float64)
    params_y = np.empty(total, dtype=np.float64)

    for j in range(rows):
        for i in range(cols):
            idx = i + j * cols
            dx = (i - left_n) * (right_step if i > left_n else left_step)
            dy = (j - down_n) * (up_step if j > down_n else down_step)
            params_x[idx] = start_x + dx
            params_y[idx] = start_y + dy

    # ВАЖНО: возвращаем все 4 значения
    return params_x, params_y, cols, rows


def build_parameter_grid_from_config(config):
    def_sys = config["defaultSystem"]
    grid = config["grid"]

    # Получаем имена параметров из конфига
    name_x = grid["first"]["name"]
    name_y = grid["second"]["name"]

    # Исправлено извлечение параметров: берем значения из defaultSystem
    params_x, params_y, cols, rows = _generate_parameters_2d(
        start_x=float(def_sys[name_x]),
        start_y=float(def_sys[name_y]),
        up_n=int(grid["second"]["up_n"]),
        down_n=int(grid["second"]["down_n"]),
        left_n=int(grid["first"]["left_n"]),
        right_n=int(grid["first"]["right_n"]),
        up_step=float(grid["second"]["up_step"]),
        down_step=float(grid["second"]["down_step"]),
        left_step=float(grid["first"]["left_step"]),
        right_step=float(grid["first"]["right_step"])
    )
    return params_x, params_y, cols, rows


def reshape_map(values, cols, rows):
    return np.asarray(values, dtype=float).reshape(rows, cols)


def extract_horizontal_scan(kneading_map_flat, config, row_index=None):
    params_x, params_y, cols, rows = build_parameter_grid_from_config(config)
    arr2d = reshape_map(kneading_map_flat, cols, rows)
    if row_index is None: row_index = int(config["grid"]["second"]["down_n"])
    seq_len = int(config["kneadings_pendulums"]["kneadings_end"]) - int(
        config["kneadings_pendulums"]["kneadings_start"]) + 1
    return [ScanPoint(i, row_index, float(params_x[i + row_index * cols]), float(params_y[i + row_index * cols]),
                      float(arr2d[row_index, i]), decode_base25_weighted(float(arr2d[row_index, i]), seq_len)) for i in
            range(cols)]


def find_code_transitions_on_scan(scan):
    return [(i, i + 1) for i in range(len(scan) - 1) if
            scan[i].raw_value >= 0 and scan[i + 1].raw_value >= 0 and scan[i].code != scan[i + 1].code]


def integrate_trajectory(y0, gamma, lam, k, dt, n_steps, stride=1, infinity_threshold=1e6):
    y = np.array(y0, dtype=float).copy()
    traj, t = np.empty((n_steps + 1, 4)), np.empty(n_steps + 1)
    traj[0], t[0] = y, 0.0
    for step in range(1, n_steps + 1):
        for _ in range(stride): y = rk4_step(y, dt, gamma, lam, k)
        traj[step], t[step] = y, step * dt * stride
        if np.any(np.abs(y) > infinity_threshold): return t[:step + 1], traj[:step + 1]
    return t, traj


def swap_pendulums(eq: np.ndarray) -> np.ndarray:
    return np.array([eq[2], eq[3], eq[0], eq[1]], dtype=float)


def find_candidate_equilibria(gamma, lam, k):
    eqs = [np.asarray(eq, dtype=float) for eq in find_equilibria_pendulum(gamma, k)]
    candidates = []
    for eq in eqs:
        info = equilibrium_type(eq, gamma, lam, k)
        is_diag = np.isclose(eq[0], eq[2], atol=1e-9) and np.isclose(eq[1], 0.0, atol=1e-9) and np.isclose(eq[3], 0.0,
                                                                                                           atol=1e-9)
        has_partner = any(np.linalg.norm(swap_pendulums(eq) - other) < 1e-8 for other in eqs)
        candidates.append(
            EquilibriumCandidate(eq, int(info["nU"]), int(info["nS"]), int(info["nC"]), np.asarray(info["eigvals"]),
                                 is_diag or has_partner))
    return candidates


def log_distance_curve(traj, eq):
    return np.log10(np.linalg.norm(traj - eq[None, :], axis=1) + EPS_LOG)


def split_distance_curves(traj, eq):
    return np.log10(np.linalg.norm(traj[:, [0, 2]] - eq[None, [0, 2]], axis=1) + EPS_LOG), \
        np.log10(np.linalg.norm(traj[:, [1, 3]] - eq[None, [1, 3]], axis=1) + EPS_LOG)


def pick_best_equilibrium(traj, t, candidates, source_eq, prefer_symmetric=False):
    working = [c for c in candidates if c.is_symmetric] if prefer_symmetric else list(candidates)
    best_eq, best_meta, best_curve, best_min_val, best_t, best_idx = None, None, None, None, None, None
    cutoff = int(0.1 * len(traj))
    for c in working:
        if np.linalg.norm(c.point - source_eq) < 1e-6: continue
        curve = log_distance_curve(traj, c.point)
        idx = int(np.argmin(curve[cutoff:])) + cutoff
        if best_min_val is None or curve[idx] < best_min_val:
            best_eq, best_meta, best_curve, best_min_val, best_t, best_idx = c.point, {"nU": c.nU, "nS": c.nS,
                                                                                       "nC": c.nC, "eigvals": c.eigvals,
                                                                                       "is_symmetric": c.is_symmetric}, curve, float(
                curve[idx]), float(t[idx]), idx
    return (best_eq, best_meta, best_curve, best_min_val, best_t, best_idx) if best_eq is not None else \
        (source_eq, {"nU": -1, "nS": -1, "nC": -1, "eigvals": np.array([]), "is_symmetric": False},
         log_distance_curve(traj, source_eq), float(np.min(log_distance_curve(traj, source_eq))), 0.0, 0)


def pick_common_separator_equilibrium(probes, candidates, cutoff_fraction=0.1, threshold=-1.0):
    best_eq, best_meta, best_score = None, None, None
    for c in candidates:
        if any(np.linalg.norm(c.point - p.source_eq) < 1e-5 for p in probes): continue
        mins = [float(np.min(log_distance_curve(p.trajectory[int(cutoff_fraction * len(p.trajectory)):], c.point))) for
                p in probes if len(p.trajectory) > 10]
        if len(mins) == len(probes) and (best_score is None or max(mins) < best_score):
            best_score, best_eq, best_meta = max(mins), c.point, {"nU": c.nU, "nS": c.nS, "nC": c.nC,
                                                                  "eigvals": c.eigvals, "is_symmetric": c.is_symmetric,
                                                                  "score": max(mins), "mins": mins}
    return (np.asarray(best_eq), best_meta) if best_eq is not None and best_score <= threshold else (None, None)


def collect_probe_candidate_equilibria(probes, tol=1e-8):
    collected = []
    for p in probes:
        for c in p.candidate_eqs:
            if not any(np.linalg.norm(c.point - old.point) < tol for old in collected): collected.append(c)
    return collected


def _build_params_for_probe(config, x, y):
    d, px, py = config["defaultSystem"], config["grid"]["first"]["name"], config["grid"]["second"]["name"]
    p = {"gamma": d["gamma"], "lambda": d["lambda"], "k": d["k"], px: x, py: y}
    return p["gamma"], p["lambda"], p["k"]


def _make_probe_result(
        label: str,
        side: str,
        config: Dict[str, Any],
        x_probe: float,
        y_probe: float,
        dt_traj: float,
        n_steps_traj: int,
        stride_traj: int,
        prefer_symmetric: bool,
        prev_eq: Optional[np.ndarray] = None,
        prev_dir: Optional[np.ndarray] = None,
) -> ProbeResult:
    gamma, lam, k = _build_params_for_probe(config, x_probe, y_probe)
    candidate_eqs = find_candidate_equilibria(gamma=gamma, lam=lam, k=k)

    # --- ИСПРАВЛЕННЫЙ БЛОК ---
    # Получаем настройки и удаляем 'base_params', если он там есть
    sep_cfg = config.get("separatrix_init", {}).copy()
    if "base_params" in sep_cfg:
        del sep_cfg["base_params"]
    # -------------------------

    # Теперь передаем очищенный словарь sep_cfg
    src_eq, init, u_dir, b_id = build_separatrix_init_for_point(
        gamma=gamma,
        lam=lam,
        k=k,
        **sep_cfg,  # используем очищенный словарь
        prev_eq=prev_eq,
        ref_unstable_dir=prev_dir,
    )

    t, traj = integrate_trajectory(
        y0=init,
        gamma=gamma,
        lam=lam,
        k=k,
        dt=dt_traj,
        n_steps=n_steps_traj,
        stride=stride_traj,
    )

    best_eq, best_meta, best_curve, best_min_val, best_t, best_idx = pick_best_equilibrium(
        traj=traj,
        t=t,
        candidates=candidate_eqs,
        source_eq=src_eq,  # исправлено на src_eq (source_eq в новой версии)
        prefer_symmetric=prefer_symmetric,
    )

    return ProbeResult(
        label=label,
        side=side,
        x_param=float(x_probe),
        y_param=float(y_probe),
        params=(gamma, lam, k),
        source_eq=np.asarray(src_eq, dtype=float),
        init_point=np.asarray(init, dtype=float),
        unstable_dir=np.asarray(u_dir, dtype=float),
        branch_id=int(b_id),
        trajectory=np.asarray(traj, dtype=float),
        time=np.asarray(t, dtype=float),
        closest_eq=np.asarray(best_eq, dtype=float),
        closest_eq_meta=best_meta,
        log_distance_to_best_eq=np.asarray(best_curve, dtype=float),
        best_eq_min_log10=float(best_min_val),
        best_eq_argmin_t=float(best_t),
        best_eq_argmin_index=int(best_idx),
        candidate_eqs=list(candidate_eqs),
    )


def analyze_separatrix_transition_v2(config, kneading_map_flat, output_dir, row_index=None, transition_number=0,
                                     closeness=0.5, dt_traj=None, n_steps_traj=30000, stride_traj=1,
                                     prefer_symmetric_equilibria=False):
    os.makedirs(output_dir, exist_ok=True)
    scan = extract_horizontal_scan(kneading_map_flat, config, row_index)
    trans = find_code_transitions_on_scan(scan)
    l_sc, r_sc = scan[trans[transition_number][0]], scan[trans[transition_number][1]]
    x_est, y_val = 0.5 * (l_sc.x + r_sc.x), l_sc.y
    dt = dt_traj if dt_traj else config["kneadings_pendulums"]["dt"]
    probes, prev_eq, prev_dir = [], None, None
    for label, side in [("before", "left"), ("after", "right")]:
        pr = _make_probe_result(label, side, config, l_sc.x if side == "left" else r_sc.x, y_val, dt, n_steps_traj,
                                stride_traj, prefer_symmetric_equilibria, prev_eq, prev_dir)
        probes.append(pr);
        prev_eq, prev_dir = pr.source_eq, pr.unstable_dir
    common_c = collect_probe_candidate_equilibria(probes)
    ceq, cmeta = pick_common_separator_equilibrium(probes, common_c)
    res = TransitionAnalysisResult(l_sc.j, trans[transition_number][0], trans[transition_number][1], l_sc, r_sc, x_est,
                                   y_val, probes,
                                   [{"point": c.point, "nU": c.nU, "nS": c.nS, "nC": c.nC, "eigvals": c.eigvals,
                                     "is_symmetric": c.is_symmetric} for c in common_c], output_dir, ceq, cmeta)
    save_transition_report(res);
    make_transition_plots_v2(config, kneading_map_flat, res);
    return res


def _to_jsonable(x: Any) -> Any:
    # Если это словарь, обрабатываем рекурсивно
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # Если это список или кортеж, обрабатываем каждый элемент
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # Массивы numpy превращаем в списки и снова прогоняем через обработку
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())

    # ГЛАВНОЕ: обрабатываем комплексные числа (Python complex и NumPy complex)
    if isinstance(x, (complex, np.complexfloating)):
        return {"real": float(x.real), "imag": float(x.imag)}

    # Остальные типы приводим к стандартным Python-типам
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)

    return x


def save_transition_report(result):
    with open(os.path.join(result.output_dir, "transition_report.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(asdict(result)), f, ensure_ascii=False, indent=2)


def make_transition_plots_v2(config: Dict[str, Any], kneading_map_flat: np.ndarray,
                             result: TransitionAnalysisResult) -> None:
    # 0. Подготовка данных сетки и среза
    params_x, params_y, cols, rows = build_parameter_grid_from_config(config)
    arr2d = reshape_map(kneading_map_flat, cols, rows)
    scan = extract_horizontal_scan(kneading_map_flat, config, result.row_index)

    x_min, x_max = float(np.min(params_x)), float(np.max(params_x))
    y_min, y_max = float(np.min(params_y)), float(np.max(params_y))

    # --- ЛОГИКА ДИНАМИЧЕСКИХ ЗВЕЗД ---
    #  Собираем координаты только для той части траекторий, которая РЕАЛЬНО рисуется
    plotted_phi1 = []
    plotted_phi2 = []
    for p in result.probes:
        mask = p.time <= 50.0
        plotted_phi1.append(p.trajectory[mask, 0])
        plotted_phi2.append(p.trajectory[mask, 2])

    all_phi1 = np.concatenate(plotted_phi1)
    all_phi2 = np.concatenate(plotted_phi2)

    phi1_range = (all_phi1.min(), all_phi1.max())
    phi2_range = (all_phi2.min(), all_phi2.max())

    # Запас (padding), чтобы видеть равновесия чуть за пределами траектории
    pad = 1.5

    o1_base = result.probes[0].source_eq
    o2_base = swap_pendulums(o1_base)

    def get_needed_shifts(base_val, v_min, v_max):
        """Вычисляет список n*2pi, попадающих в [v_min-pad, v_max+pad]"""
        n_start = int(np.floor((v_min - pad - base_val) / (2 * np.pi)))
        n_end = int(np.ceil((v_max + pad - base_val) / (2 * np.pi)))
        return [n * 2 * np.pi for n in range(n_start, n_end + 1)]

    unique_vals, inverse_indices = np.unique(arr2d, return_inverse=True)
    N_unique = len(unique_vals)

    # Нормализуем индексы на отрезок [0, 1] для правильного масштабирования в imshow
    if N_unique > 1:
        plot_arr2d = (inverse_indices / (N_unique - 1)).reshape(arr2d.shape)
    else:
        plot_arr2d = np.zeros_like(arr2d)

    cmap = set_random_color_map()
    # 1. Карта параметров
    plt.figure(figsize=(8, 6))
    # Рисуем карту по категориальным индексам plot_arr2d и жестко фиксируем vmin/vmax
    plt.imshow(plot_arr2d, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max], cmap=cmap, vmin=0,vmax=1)
    plt.axhline(result.y_scan_value, linestyle="--", color="white", alpha=0.5)
    plt.scatter([result.x_boundary_estimate], [result.y_scan_value], s=100, marker="x", color="red", label="исследуемый переход")
    plt.title("Карта параметров и выбранный переход")
    plt.xlabel(config["grid"]["first"].get("caption", "k"))
    plt.ylabel(config["grid"]["second"].get("caption", "gamma"))
    plt.savefig(os.path.join(result.output_dir, "01_parameter_map_real_axes.png"), dpi=250)
    plt.close()

    # 1.1 Карта параметров
    plt.figure(figsize=(8, 6))
    # Рисуем карту по категориальным индексам plot_arr2d и жестко фиксируем vmin/vmax
    plt.imshow(plot_arr2d, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max], cmap=cmap, vmin=0,vmax=1)
    plt.title("Карта параметров")
    plt.xlabel(config["grid"]["first"].get("caption", "k"))
    plt.ylabel(config["grid"]["second"].get("caption", "gamma"))
    plt.savefig(os.path.join(result.output_dir, "01_1_parameter_map_real_axes.png"), dpi=250)
    plt.close()

    # 2. Горизонтальный срез
    plt.figure(figsize=(11, 4))
    plt.plot([p.x for p in scan], [p.raw_value for p in scan], color="black", marker="o", markersize=3,
             label="значение нидинга")
    plt.axvline(result.x_boundary_estimate, color="red", linestyle="--", linewidth=2, label="смена кода")
    plt.title("Горизонтальный срез карты нидингов")
    plt.ylabel("raw kneading value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(result.output_dir, "02_horizontal_scan.png"), dpi=250)
    plt.close()

    # 3. Набор 4 проекций 4D-траекторий
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    proj = [((0, 2), r"$(\phi_1, \phi_2)$", r"$\phi_1$", r"$\phi_2$"),
            ((1, 3), r"$(v_1, v_2)$", r"$v_1$", r"$v_2$"),
            ((0, 1), r"$(\phi_1, v_1)$", r"$\phi_1$", r"$v_1$"),
            ((2, 3), r"$(\phi_2, v_2)$", r"$\phi_2$", r"$v_2$")]

    colors = {"before": "tab:orange", "after": "tab:green"}

    for ax, ((ii, jj), title, xl, yl) in zip(axes.ravel(), proj):
        # Определяем, по каким осям нужно делать сдвиги на 2pi (только для углов 0 и 2)
        shifts_i = get_needed_shifts(o1_base[ii], phi1_range[0] if ii == 0 else phi2_range[0],
                                     phi1_range[1] if ii == 0 else phi2_range[1]) if ii in [0, 2] else [0]
        shifts_j = get_needed_shifts(o1_base[jj], phi1_range[0] if jj == 0 else phi2_range[0],
                                     phi1_range[1] if jj == 0 else phi2_range[1]) if jj in [0, 2] else [0]

        # Отрисовка звезд равновесий
        first_star = True
        for si in shifts_i:
            for sj in shifts_j:
                p1_i, p1_j = o1_base[ii] + si, o1_base[jj] + sj
                p2_i, p2_j = o2_base[ii] + si, o2_base[jj] + sj

                ax.scatter(p1_i, p1_j, marker='*', color='darkgrey', s=100, edgecolor='white', lw=0.3, zorder=10,
                           label='равновесие $O_1$' if first_star else "")
                ax.scatter(p2_i, p2_j, marker='*', color='black', s=100, edgecolor='white', lw=0.3, zorder=10,
                           label='равновесие $O_2$' if first_star else "")
                first_star = False

        # Отрисовка траекторий
        for p in result.probes:
            mask = p.time <= 50.0
            ax.plot(p.trajectory[mask, ii], p.trajectory[mask, jj], color=colors[p.label], lw=2, alpha=0.9,
                    label=p.label, zorder=5)
            # Начало (ромб) и конец (крестик)
            ax.scatter(p.trajectory[0, ii], p.trajectory[0, jj], color=colors[p.label], marker="D", s=40, ec="k",
                       zorder=11)
            ax.scatter(p.trajectory[mask][-1, ii], p.trajectory[mask][-1, jj], color=colors[p.label], marker="x", s=50,
                       zorder=12)

        ax.set_title(title);
        ax.set_xlabel(xl);
        ax.set_ylabel(yl);
        ax.grid(True, alpha=0.3)

        # Фиксируем лимиты осей, чтобы "лишние" звезды не отдаляли камеру
        if ii in [0, 2]: ax.set_xlim(phi1_range[0] - pad, phi1_range[1] + pad) if ii == 0 else ax.set_xlim(
            phi2_range[0] - pad, phi2_range[1] + pad)
        if jj in [0, 2]: ax.set_ylim(phi1_range[0] - pad, phi1_range[1] + pad) if jj == 0 else ax.set_ylim(
            phi2_range[0] - pad, phi2_range[1] + pad)
        if (ii, jj) == (0, 2):
            ax.set_xlim(1.0, 2.0)
        if (ii, jj) == (0, 1):
            # Жесткое ограничение оси x для проекции (phi1, v1) от 1.0 до 5.0
            ax.set_xlim(1.0, 2.0)

    plt.tight_layout();
    plt.savefig(os.path.join(result.output_dir, "03_phase_projections_4d.png"), dpi=250);
    plt.close()

    # 4. Zoom графики (используем те же умные звезды)
    t_zoom = 50.0
    for ztype in ["angles", "velocities"]:
        plt.figure(figsize=(8, 8))
        ii, jj = (0, 2) if ztype == "angles" else (1, 3)
        plt.title(
            r"Увеличенная проекция $(\phi_1, \phi_2)$" if ztype == "angles" else r"Увеличенная проекция $(v_1, v_2)$")

        # Звезды в зуме
        shifts_i = get_needed_shifts(o1_base[ii], phi1_range[0] if ii == 0 else phi2_range[0],
                                     phi1_range[1] if ii == 0 else phi2_range[1]) if ii in [0, 2] else [0]
        shifts_j = get_needed_shifts(o1_base[jj], phi1_range[0] if jj == 0 else phi2_range[0],
                                     phi1_range[1] if jj == 0 else phi2_range[1]) if jj in [0, 2] else [0]
        for si in shifts_i:
            for sj in shifts_j:
                plt.scatter(o1_base[ii] + si, o1_base[jj] + sj, marker='*', color='darkgrey', s=150, ec='w', zorder=10)
                plt.scatter(o2_base[ii] + si, o2_base[jj] + sj, marker='*', color='black', s=150, ec='w', zorder=10)

        for p in result.probes:
            mask = p.time <= t_zoom
            plt.plot(p.trajectory[mask, ii], p.trajectory[mask, jj], color=colors[p.label], lw=2.5, label=p.label)

        # Жесткие границы для зума
        if ztype == "angles":
            plt.xlim(1.5, 2.0);
            plt.ylim(10, 50)
            plt.xlabel(r"$\phi_1$");
            plt.ylabel(r"$\phi_2$")
        else:
            plt.xlim(-0.1, 0.1);
            plt.ylim(3, 5)
            plt.xlabel(r"$v_1$");
            plt.ylabel(r"$v_2$")

        plt.grid(True, alpha=0.3);
        plt.legend();
        plt.savefig(os.path.join(result.output_dir, f"03_zoom_{ztype}.png"), dpi=250);
        plt.close()

    # 5. Расстояния (без изменений)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for p in result.probes:
        mask = p.time <= 50.0
        eq = result.common_eq if result.common_eq is not None else p.closest_eq
        axes[0].plot(p.time[mask], log_distance_curve(p.trajectory[mask], eq), color=colors[p.label], label=p.label)
        a_c, v_c = split_distance_curves(p.trajectory[mask], eq)
        axes[1].plot(p.time[mask], a_c, color=colors[p.label]);
        axes[2].plot(p.time[mask], v_c, color=colors[p.label])
    plt.tight_layout();
    plt.savefig(os.path.join(result.output_dir, "05_distance.png"), dpi=250);
    plt.close()


def save_human_summary(result):
    with open(os.path.join(result.output_dir, "transition_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Transition between {result.left_scan_point.code} and {result.right_scan_point.code}\n"
                f"Boundary x ~ {result.x_boundary_estimate}\n")


__all__ = ["analyze_separatrix_transition_v2"]