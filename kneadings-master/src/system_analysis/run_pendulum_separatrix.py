# src/system_analysis/run_pendulum_separatrix.py
# -*- coding: utf-8 -*-

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib.eq_finder.systems_fun as sf
from lib.eq_finder.TwoCoupledPendulums import TwoPendulums
from src.mapping.events_pendulums import detect_event_0_7


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

OUT_DIR = os.path.join(PROJECT_ROOT, "output_separatrix")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] Saving results to: {OUT_DIR}")


# Находит все состояния равновесия системы
def find_all_equilibria_4d(sys_obj, ps,
                           n_samples=1500, n_iters=3, eps=1e-10,
                           vel_bounds=(-0.5, 0.5)):

    bounds = [ # мягкие границы для начального поиска (можно немного выходить)
        (-0.1, 2 * np.pi + 0.1), # fi1 может быть от -0.1 до 2π+0.1
        (vel_bounds[0], vel_bounds[1]), # v1 в заданных границах
        (-0.1, 2 * np.pi + 0.1), # fi2 может быть от -0.1 до 2π+0.1
        (vel_bounds[0], vel_bounds[1]), # v2 в заданных границах
    ]

    borders = [ #жесткие границы (нельзя выходить ни при каких условиях)
        (-1e-15, 2 * np.pi + 1e-15),
        vel_bounds,
        (-1e-15, 2 * np.pi + 1e-15),
        vel_bounds,
    ]

    rhs = lambda X: sys_obj.FullSystem(X) #Функция, которая вычисляет правые части системы
    rhs_jac = lambda X: np.array(sys_obj.Jac(X), dtype=float) #Матрица частных производных правых частей

    #Создание оптимизатора. Гарантированно находит все равновесия (если они есть)
    eq_finder = sf.ShgoEqFinder(n_samples, n_iters, eps)

    return sf.findEquilibria(
        rhs=rhs,
        rhsJac=rhs_jac,
        eqRhs=rhs,   # для проверки равновесий
        eqJac=rhs_jac,   #  для анализа
        embedInPhaseSpace=lambda X: X,
        bounds=bounds,
        borders=borders,
        optMethod=eq_finder,  # метод оптимизации (SHGO)
        ps=ps,    # точность вычислений
    )


#фильтрует список всех найденных равновесий, оставляя только седло-фокусы с одномерным неустойчивым многообразием
def pick_saddle_foci(eqs, ps):
    return [eq for eq in eqs if sf.is4DSaddleFocusWith1dU(eq, ps)]


#вычисление сепаратрис
def compute_separatrices(sys_obj, eq, ps, max_time=300.0):

    rhs = lambda X: sys_obj.FullSystem(X)

    return sf.computeSeparatrices(
        eq=eq, #равновесие
        rhs=rhs,
        ps=ps,
        maxTime=max_time,
        condition=sf.pickBothSeparatrices, #указывает, что нужно взять обе ветви неустойчивого многообразия.
        tSkip=0.0,
        listEvents=None
    )


#Вспомогательная функция сохранения рисунка
def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_separatrices(seps, eq):

    eq = np.asarray(eq)

    # (fi1, v1)
    plt.figure()
    for s in seps:
        s = np.asarray(s)
        plt.plot(s[:, 0], s[:, 1])
    plt.scatter(eq[0], eq[1], c="k")
    plt.xlabel("fi1")
    plt.ylabel("v1")
    save_fig("sep_phi1_v1.png")

    # (fi2, v2)
    plt.figure()
    for s in seps:
        s = np.asarray(s)
        plt.plot(s[:, 2], s[:, 3])
    plt.scatter(eq[2], eq[3], c="k")
    plt.xlabel("fi2")
    plt.ylabel("v2")
    save_fig("sep_phi2_v2.png")

    # (fi1, fi2)
    plt.figure()
    for s in seps:
        s = np.asarray(s)
        plt.plot(s[:, 0], s[:, 2])
    plt.scatter(eq[0], eq[2], c="k")
    plt.xlabel("fi1")
    plt.ylabel("fi2")
    save_fig("sep_phi1_phi2.png")

def encode_kneading_from_traj(traj_4d, max_events=200, start_from=0):
    """
    Возвращает список символов 0..7 по траектории.
    traj_4d: shape (T,4) [fi1, v1, fi2, v2]
    """
    traj_4d = np.asarray(traj_4d, dtype=float)

    symbols = []
    evt_count = 0

    for i in range(1, len(traj_4d)):
        ev = detect_event_0_7(traj_4d[i-1], traj_4d[i])
        if ev >= 0:
            if evt_count >= start_from:
                symbols.append(int(ev))   # <-- важно: int
                if len(symbols) >= max_events:
                    break
            evt_count += 1

    return symbols


def encode_kneadings_for_separatrices(seps, max_events=200, start_from=0):
    """
    Для каждой ветви сепаратрисы возвращает:
    - список символов (0..7)
    - строку "0 5 1 4 ..."
    """
    kneadings = []
    kneadings_str = []

    for s in seps:
        knd = encode_kneading_from_traj(s, max_events=max_events, start_from=start_from)
        kneadings.append(knd)
        kneadings_str.append(" ".join(map(str, knd)))

    return kneadings, kneadings_str



def main():

    gamma = 0.97
    lam = 0.2
    k = 0.06

    sys_obj = TwoPendulums(gamma, lam, k)
    ps = sf.STD_PRECISION

    eqs = find_all_equilibria_4d(sys_obj, ps)
    print(f"All equilibria found: {len(eqs)}")

    sadfocs = pick_saddle_foci(eqs, ps)
    print(f"Saddle-foci found: {len(sadfocs)}")

    if not sadfocs:
        raise RuntimeError("No saddle-focus found")

    eq = sadfocs[0]
    print("Chosen saddle-focus:", eq.coordinates)
    print("Eigenvalues:", eq.eigenvalues)

    seps, times = compute_separatrices(sys_obj, eq, ps)
    print("Separatrix lengths:", [len(s) for s in seps])

    # --- KNEADINGS from separatrices (symbols 0..7) ---
    kneadings, kneadings_str = encode_kneadings_for_separatrices(seps, max_events=200, start_from=0)

    print("\n=== KNEADING SEQUENCES (symbols 0..7) ===")
    for bi, knd in enumerate(kneadings):
        preview_n = min(80, len(knd))
        preview = " ".join(map(str, knd[:preview_n]))
        print(f"\nBranch {bi}:")
        print(f"  length = {len(knd)}")
        print(f"  preview(first {preview_n}) = {preview}")
        print(f"  FULL sequence:")
        print(f"  {kneadings_str[bi]}")

    # ===== save NPZ =====
    out_npz_path = os.path.join(OUT_DIR, f"sep_gamma{gamma}_lam{lam}_k{k}.npz")

    np.savez(
        out_npz_path,
        gamma=gamma, lam=lam, k=k,
        eq_coords=np.asarray(eq.coordinates, dtype=float),
        eigvals=np.asarray(eq.eigenvalues),
        separatrix_0=np.asarray(seps[0], dtype=float),
        separatrix_1=np.asarray(seps[1], dtype=float),
        kneading_0=np.asarray(kneadings[0], dtype=np.int32),
        kneading_1=np.asarray(kneadings[1], dtype=np.int32),
    )
    print("NPZ saved:", out_npz_path)

    out_txt_path = os.path.join(OUT_DIR, f"sep_gamma{gamma}_lam{lam}_k{k}_kneadings.txt")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for i in range(len(kneadings_str)):
            f.write(f"branch {i} (len={len(kneadings[i])}):\n")
            f.write(kneadings_str[i] + "\n\n")
    print("Kneadings saved:", out_txt_path)


if __name__ == "__main__":
    main()
