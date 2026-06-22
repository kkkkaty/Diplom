import os
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize_scalar

# === ПАРАМЕТРЫ СИСТЕМЫ ===
k_val = 0.0698
gamma_val = 0.9696
# Расширяем диапазон поиска lambda
lam_bounds = (0.10, 0.40)

# Путь к вашему JSON-файлу для автоподбора зацепок
json_path = r"C:/Lobach4/tu/kneadings-master1/output/manual_transition_check/transition_report.json"

if not os.path.exists(json_path):
    print(f"❌ ОШИБКА: Файл {json_path} не найден!")
    print("Сначала запустите manual_transition_check.py для генерации отчета!")
    exit()


def system(t, y, gamma, lam, k):
    phi1, v1, phi2, v2 = y
    return [
        v1,
        gamma - lam * v1 - np.sin(phi1) + k * np.sin(phi2 - phi1),
        v2,
        gamma - lam * v2 - np.sin(phi2) + k * np.sin(phi1 - phi2)
    ]


def event_v1_zero(t, y, gamma, lam, k):
    return y[1]


event_v1_zero.terminal = True
event_v1_zero.direction = 1


def angle_diff(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def poincare_map(u, gamma, lam, k):
    phi1, phi2, v2 = u
    y0 = np.array([phi1, 0.0, phi2, v2], dtype=float)

    sol_init = solve_ivp(system, [0.0, 0.04], y0, args=(gamma, lam, k), rtol=1e-10, atol=1e-10)
    y_start = sol_init.y[:, -1]

    sol = solve_ivp(system, [0.04, 100.0], y_start, args=(gamma, lam, k),
                    events=event_v1_zero, rtol=1e-10, atol=1e-10)

    if not sol.t_events[0].size:
        return None, None

    y_return = sol.y[:, -1]
    phi1_ret = np.mod(y_return[0] + np.pi, 2 * np.pi) - np.pi
    phi2_ret = np.mod(y_return[2] + np.pi, 2 * np.pi) - np.pi
    v2_ret = y_return[3]
    T = sol.t_events[0][0] + 0.04

    return np.array([phi1_ret, phi2_ret, v2_ret]), T


def poincare_residual(u, gamma, lam, k):
    p_u, _ = poincare_map(u, gamma, lam, k)
    if p_u is None:
        return np.array([1e3, 1e3, 1e3])
    res = np.zeros(3)
    res[0] = angle_diff(p_u[0], u[0])
    res[1] = angle_diff(p_u[1], u[1])
    res[2] = p_u[2] - u[2]
    return res


def compute_poincare_jacobian(u_star, gamma, lam, k, eps=1e-7):
    J = np.zeros((3, 3))
    for j in range(3):
        du = np.zeros(3)
        du[j] = eps
        p_plus, _ = poincare_map(u_star + du, gamma, lam, k)
        p_minus, _ = poincare_map(u_star - du, gamma, lam, k)
        if p_plus is None or p_minus is None:
            return None
        d_phi1 = angle_diff(p_plus[0], p_minus[0])
        d_phi2 = angle_diff(p_plus[1], p_minus[1])
        d_v2 = p_plus[2] - p_minus[2]
        J[:, j] = np.array([d_phi1, d_phi2, d_v2]) / (2 * eps)
    return J


# === 3. ПОИСК СЕДЛО-ФОКУСА O1 ===

def find_saddle_focus_O1(gamma, lam, k):
    from scipy.optimize import fsolve
    def equations(x):
        p1, p2 = x
        return [
            gamma - np.sin(p1) + k * np.sin(p2 - p1),
            gamma - np.sin(p2) + k * np.sin(p1 - p2)
        ]

    guesses = [(1.5, 3.0), (1.5, 2.0)]
    for g in guesses:
        sol = fsolve(equations, g)
        if sol[0] < sol[1]:  # правило phi1 < phi2
            return np.array([sol[0], 0.0, sol[1], 0.0], dtype=float)
    return None


# === 4. АВТОМАТИЧЕСКИЙ СБОР ПЕРЕСЕЧЕНИЙ ДЛЯ СТАРТА ===

def get_crossings_from_traj(traj, t_arr):
    crossings = []
    for i in range(1, len(traj)):
        v1_prev = traj[i - 1][1]
        v1_curr = traj[i][1]
        if v1_prev < 0 and v1_curr > 0:
            frac = -v1_prev / (v1_curr - v1_prev)
            p1 = traj[i - 1][0] + frac * (traj[i][0] - traj[i - 1][0])
            p2 = traj[i - 1][2] + frac * (traj[i][2] - traj[i - 1][2])
            v2 = traj[i - 1][3] + frac * (traj[i][3] - traj[i - 1][3])
            t_c = t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1])

            p1 = np.mod(p1 + np.pi, 2 * np.pi) - np.pi
            p2 = np.mod(p2 + np.pi, 2 * np.pi) - np.pi
            crossings.append((t_c, np.array([p1, p2, v2])))
    return crossings


# Загружаем отчет ОДИН раз при запуске программы
with open(json_path, "r", encoding="utf-8") as f:
    report = json.load(f)

before_probe = [p for p in report["probes"] if p["label"] == "before"][0]
before_traj = np.array(before_probe["trajectory"])
before_time = np.array(before_probe["time"])

# Находим все зацепки-пересечения
all_crossings = get_crossings_from_traj(before_traj, before_time)


# === 5. ЦЕЛЕВАЯ ФУНКЦИЯ ДЛЯ МИНИМИЗАЦИИ ===

def get_heteroclinic_distance(lam):
    """
    Для заданного lambda находит седловой цикл и возвращает
    минимальное расстояние от его неустойчивых ветвей до седла O1.
    """
    O1 = find_saddle_focus_O1(gamma_val, lam, k_val)
    if O1 is None:
        return 10.0

    # Ищем седловой цикл, перебирая зацепки из середины спирали
    sol_root = None
    saddle_cycle_eigvec = None

    # Чтобы оптимизация была быстрой, проверим только надежные витки из середины
    candidates = list(enumerate(all_crossings))
    candidates.reverse()

    for idx, (t_c, u_guess) in candidates:
        if idx < 100 or idx > len(all_crossings) - 20:
            continue  # берем надежный "хвост" спирали до разлета

        sol = root(poincare_residual, u_guess, args=(gamma_val, lam, k_val), method='hybr', tol=1e-8)
        if sol.success:
            u_star = sol.x
            res_val = np.linalg.norm(poincare_residual(u_star, gamma_val, lam, k_val))
            if res_val < 1e-5:
                J = compute_poincare_jacobian(u_star, gamma_val, lam, k_val)
                if J is not None:
                    eigvals, eigvecs = np.linalg.eig(J)
                    unstable_indices = np.where(np.abs(eigvals) > 1.001)[0]
                    if len(unstable_indices) == 1:
                        sol_root = sol
                        unstable_idx = unstable_indices[0]
                        saddle_cycle_eigvec = np.real(eigvecs[:, unstable_idx])
                        break

    if sol_root is None:
        print(f"  [λ = {lam:.5f}] -> Седловой цикл не найден")
        return 10.0  # Штраф

    u_star = sol_root.x
    v_u = saddle_cycle_eigvec / np.linalg.norm(saddle_cycle_eigvec)

    # Интегрируем ветви неустойчивого многообразия Wu
    dists = []
    for direction in [+1, -1]:
        u_start = u_star + direction * 1e-4 * v_u
        y0_start = np.array([u_start[0], 0.0, u_start[1], u_start[2]])

        sol = solve_ivp(system, [0, 200], y0_start, args=(gamma_val, lam, k_val), rtol=1e-9, atol=1e-9, max_step=0.05)

        # Вычисляем минимальное расстояние до O1
        diffs = sol.y - O1[:, None]
        diffs[0] = angle_diff(sol.y[0], O1[0])
        diffs[2] = angle_diff(sol.y[2], O1[2])
        d_min = np.min(np.linalg.norm(diffs, axis=0))
        dists.append(d_min)

    min_d = min(dists)
    print(f"  [λ = {lam:.5f}] -> Цикл найден! Мин. расстояние до O1 = {min_d:.2e} рад")
    return min_d


# === 6. ЗАПУСК ОПТИМИЗАЦИИ ===
print("🔍 Ищу оптимальный lambda методом минимизации...")
res = minimize_scalar(get_heteroclinic_distance, bounds=lam_bounds, method='bounded', options={'xatol': 1e-6})

best_lam = res.x
best_dist = res.fun

print(f"\n==================================================")
print("✅ РЕЗУЛЬТАТ ПОИСКА СВЯЗИ СЕДЛОВОЙ ЦИКЛ ↔ СЕДЛО-ФОКУС:")
print(f"==================================================")
print(f"  Оптимальный lambda: {best_lam:.7f}")
print(f"  Минимальное расстояние Wu(Cycle) до O1: {best_dist:.2e} радиан")

if best_dist < 1e-3:
    print("\n🎉 УСПЕХ! Найдена точная гетероклиническая связь!")
    print("   Неустойчивое многообразие седлового цикла ложится точно на устойчивое многообразие O1.")
else:
    print("\n⚠️ Точной связи не найдено, расстояние велико.")
    print("   Попробуйте слегка сместить gamma_val и повторить поиск.")