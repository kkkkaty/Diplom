# src/system_analysis/run_pendulum_separatrix.py
# -*- coding: utf-8 -*-

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Использование Agg для работы без GUI
import matplotlib.pyplot as plt

import lib.eq_finder.systems_fun as sf
from lib.eq_finder.TwoCoupledPendulums import TwoPendulums
from src.mapping.events_pendulums import detect_event_0_7
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

OUT_DIR = os.path.join(PROJECT_ROOT, "output_separatrix")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# УПРОЩЕННЫЕ УРАВНЕНИЯ ДЛЯ ПОИСКА РАВНОВЕСИЙ (eqRhs)
# =========================================================

def simplified_equilibrium_rhs(X, gamma, k):
    """
    Функция для поиска корней. Исключает затухание и
    явно задает условия покоя для скоростей.
    """
    phi1, v1, phi2, v2 = X
    return np.array([
        v1,
        gamma - np.sin(phi1) + k * np.sin(phi2 - phi1),
        v2,
        gamma - np.sin(phi2) + k * np.sin(phi1 - phi2)
    ], dtype=float)


def simplified_equilibrium_jac(X, gamma, k):
    """
    Якобиан упрощенной системы.
    """
    phi1, v1, phi2, v2 = X
    c12 = np.cos(phi2 - phi1)

    df2_dphi1 = -np.cos(phi1) - k * c12
    df2_dphi2 = k * c12
    df4_dphi1 = k * c12
    df4_dphi2 = -np.cos(phi2) - k * c12

    return np.array([
        [0, 1, 0, 0],
        [df2_dphi1, 0, df2_dphi2, 0],
        [0, 0, 0, 1],
        [df4_dphi1, 0, df4_dphi2, 0]
    ], dtype=float)


# =========================================================
# ОСНОВНОЙ ФУНКЦИОНАЛ
# =========================================================

def find_all_equilibria_4d(sys_obj, ps, gamma, k, vel_bounds=(-1.0, 1.0)):
    from scipy.optimize import root
    import numpy as np

    phi_grid = np.linspace(0, 2 * np.pi, 10)
    found_points = []

    eq_rhs_fun = lambda X: simplified_equilibrium_rhs(X, gamma, k)
    eq_jac_fun = lambda X: simplified_equilibrium_jac(X, gamma, k)

    for p1 in phi_grid:
        for p2 in phi_grid:
            guess = [p1, 0.0, p2, 0.0]
            sol = root(eq_rhs_fun, guess, jac=eq_jac_fun, method='hybr')

            if sol.success:
                res = np.linalg.norm(eq_rhs_fun(sol.x))
                if res < 1e-9:
                    # ИСПРАВЛЕНИЕ: mod только для углов, скорости в 0
                    p = np.array(sol.x).copy()
                    p[0] = np.mod(p[0], 2 * np.pi)
                    p[2] = np.mod(p[2], 2 * np.pi)
                    p[1] = 0.0
                    p[3] = 0.0

                    if not any(np.allclose(p, fp, atol=1e-2) for fp in found_points):
                        found_points.append(p)

    class EqWrapper:
        def __init__(self, coords, sys_obj):
            self.coordinates = coords
            jac_mtx = np.array(sys_obj.Jac(coords), dtype=float)
            vals, vecs = np.linalg.eig(jac_mtx)
            idx_unstable = np.argmax(np.real(vals))
            indices = [i for i in range(len(vals)) if i != idx_unstable] + [idx_unstable]
            self.eigenvalues = vals[indices]
            self.eigvectors = vecs.T[indices]

        def getEqType(self, ps):
            eps = 1e-10
            eigvals = self.eigenvalues
            stable = [ev for ev in eigvals if ev.real < -eps]
            unstable = [ev for ev in eigvals if ev.real > eps]
            center = [ev for ev in eigvals if abs(ev.real) <= eps]
            nS, nU, nC = len(stable), len(unstable), len(center)
            isSComplex = any(abs(ev.imag) > eps for ev in stable)
            isUComplex = any(abs(ev.imag) > eps for ev in unstable)
            return [nS, nC, nU, int(isSComplex), int(isUComplex)]

    return [EqWrapper(p, sys_obj) for p in found_points]


def pick_saddle_foci(eqs, ps):
    """Фильтрует только седло-фокусы с 1D неустойчивостью"""
    return [eq for eq in eqs if sf.is4DSaddleFocusWith1dU(eq, ps)]


def compute_separatrices(sys_obj, eq, ps, max_time=300.0):
    rhs = lambda X: sys_obj.FullSystem(X)
    return sf.computeSeparatrices(
        eq=eq,
        rhs=rhs,
        ps=ps,
        maxTime=max_time,
        condition=sf.pickBothSeparatrices,
        tSkip=0.0,
        listEvents=None
    )


def encode_kneading_from_traj(traj_4d, max_events=200):
    traj_4d = np.asarray(traj_4d, dtype=float)
    symbols = []
    for i in range(1, len(traj_4d)):
        ev = detect_event_0_7(traj_4d[i - 1], traj_4d[i])
        if ev >= 0:
            symbols.append(int(ev))
            if len(symbols) >= max_events:
                break
    return symbols


def plot_separatrices(seps, eq, gamma, lam, k):
    eq = np.asarray(eq)
    fig_names = ["phi1_v1", "phi2_v2", "phi1_phi2"]
    projections = [(0, 1), (2, 3), (0, 2)]
    labels = [("fi1", "v1"), ("fi2", "v2"), ("fi1", "fi2")]

    for idx, (p_idx, p_labels) in enumerate(zip(projections, labels)):
        plt.figure(figsize=(8, 6))
        for s in seps:
            s = np.asarray(s)
            plt.plot(s[:, p_idx[0]], s[:, p_idx[1]], alpha=0.8)
        plt.scatter(eq[p_idx[0]], eq[p_idx[1]], c="black", marker="*", s=100, label="Saddle-focus", zorder=10)
        plt.xlabel(p_labels[0])
        plt.ylabel(p_labels[1])
        plt.title(fr"Separatrices: $\gamma={gamma}, \lambda={lam}, k={k}$")
        plt.grid(True, alpha=0.3)

        path = os.path.join(OUT_DIR, f"sep_{fig_names[idx]}_g{gamma}_k{k}.png")
        plt.savefig(path, dpi=300)
        plt.close()


def get_human_type(type_vec):
    nS, nC, nU, isSComp, isUComp = type_vec
    if nS == 4:
        return "Stable Focus" if isSComp else "Stable Node"
    if nU == 4:
        return "Unstable Focus" if isUComp else "Unstable Node"
    if nU == 1:
        return "Saddle-Focus (1D Unstable)" if isSComp else "Saddle (1D Unstable)"
    if nU == 2:
        return "Saddle (2D Unstable)"
    if nU == 3:
        return "Saddle-Focus (3D Unstable)"
    return f"Saddle/Other (nU={nU})"


def choose_target_saddle_focus(sadfocs, rule="phi1_lt_phi2"):
    """
    Выбирает конкретное седло из списка найденных по правилу из старого кода.
    """
    if not sadfocs:
        return None

    if rule == "phi1_lt_phi2":
        # Ищем точку, в которой phi1 < phi2
        for eq in sadfocs:
            if eq.coordinates[0] < eq.coordinates[2]:
                return eq

    # Если правило не сработало или оно другое - берем первое из списка
    return sadfocs[0]

def main():
    gamma = 0.5
    lam = 0.2
    k = 0.4322

    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ СИСТЕМЫ: gamma={gamma}, lambda={lam}, k={k}")
    print(f"{'=' * 60}")

    sys_obj = TwoPendulums(gamma, lam, k)
    ps = sf.STD_PRECISION

    eqs = find_all_equilibria_4d(sys_obj, ps, gamma, k)

    print(f"\nНайдено уникальных состояний равновесия: {len(eqs)}")
    print(f"{'№':<3} | {'Координаты (phi1, v1, phi2, v2)':<40} | {'Тип динамики':<25}")
    print("-" * 80)

    for i, eq in enumerate(eqs):
        t_vec = eq.getEqType(ps)
        h_type = get_human_type(t_vec)
        coords_str = np.array2string(eq.coordinates, precision=3, suppress_small=True)
        print(f"{i:<3} | {coords_str:<40} | {h_type}")

    # Фильтрация целевых точек
    sadfocs = pick_saddle_foci(eqs, ps)
    print(f"\nПодходящих седло-фокусов для сепаратрис: {len(sadfocs)}")

    if not sadfocs:
        print("[!] Целевые седло-фокусы не найдены.")
        return

    # Работаем с выбранным седлом
    target_eq = choose_target_saddle_focus(sadfocs, rule="phi1_lt_phi2")

    if target_eq is None:
        print("[!] Не удалось выбрать седло по правилу phi1 < phi2")
        return

    print(f"\nВыбрано седло (согласно правилу phi1 < phi2):")
    print(f"Координаты: {target_eq.coordinates}")

    # Расчет сепаратрис
    seps, times = compute_separatrices(sys_obj, target_eq, ps, max_time=500.0)
    kneadings = [encode_kneading_from_traj(s) for s in seps]

    print("\nНидинг-последовательности:")
    for i, knd in enumerate(kneadings):
        print(f"Ветка {i}: {' '.join(map(str, knd[:40]))}...")

    plot_separatrices(seps, target_eq.coordinates, gamma, lam, k)
    print(f"\n[ГОТОВО] Графики сохранены в {OUT_DIR}")


if __name__ == "__main__":
    main()