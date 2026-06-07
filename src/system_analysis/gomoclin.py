import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# === 1. ПАРАМЕТРЫ СИСТЕМЫ ===
# Начнем с области, где вы видели интересные переходы
k_val = 0.06
gamma_val = 0.97  # Попробуем из области бифуркаций
# lambda будет искать оптимизатор
lam_bounds = (0.1, 0.4)


def system(t, y, gamma, lam, k):
    phi1, v1, phi2, v2 = y
    return [
        v1,
        gamma - lam * v1 - np.sin(phi1) + k * np.sin(phi2 - phi1),
        v2,
        gamma - lam * v2 - np.sin(phi2) + k * np.sin(phi1 - phi2)
    ]


def jacobian(y, gamma, lam, k):
    phi1, v1, phi2, v2 = y
    return np.array([
        [0, 1, 0, 0],
        [-np.cos(phi1) - k * np.cos(phi2 - phi1), -lam, k * np.cos(phi2 - phi1), 0],
        [0, 0, 0, 1],
        [k * np.cos(phi1 - phi2), 0, -np.cos(phi2) - k * np.cos(phi1 - phi2), -lam]
    ])


def find_saddle_focus(gamma, lam, k):
    # Ищем состояния равновесия
    from scipy.optimize import fsolve
    def equations(x):
        p1, p2 = x
        return [
            gamma - np.sin(p1) + k * np.sin(p2 - p1),
            gamma - np.sin(p2) + k * np.sin(p1 - p2)
        ]

    # Стартовые точки для поиска (синхронные и асинхронные)
    guesses = [(1.0, 2.0), (0.5, 0.5), (2.5, 2.5)]
    equilibria = []

    for g in guesses:
        sol = fsolve(equations, g)
        # Проверка на уникальность
        is_new = True
        for eq in equilibria:
            if np.allclose(sol, eq, atol=1e-4):
                is_new = False
                break
        if is_new:
            equilibria.append([sol[0], 0.0, sol[1], 0.0])

    return equilibria


def distance_to_saddle(traj, saddle):
    """Считает минимальное расстояние от траектории до седла с учетом 2pi"""
    s_phi1, s_v1, s_phi2, s_v2 = saddle
    min_dist = 1e9

    for i in range(len(traj.t)):
        p1, v1, p2, v2 = traj.y[:, i]

        # Расстояние по фазам (учитываем цикличность)
        d_p1 = min(abs(p1 - s_phi1), 2 * np.pi - abs(p1 - s_phi1))
        d_p2 = min(abs(p2 - s_phi2), 2 * np.pi - abs(p2 - s_phi2))

        # Евклидово расстояние
        dist = np.sqrt(d_p1 ** 2 + (v1 - s_v1) ** 2 + d_p2 ** 2 + (v2 - s_v2) ** 2)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def objective_function(lam):
    """Функция, которую мы минимизируем. Возвращает расстояние до седла."""
    eqs = find_saddle_focus(gamma_val, lam, k_val)

    # Нам нужно седло-фокус (обычно это асинхронное решение при таких параметрах)
    # Берем первое попавшееся асинхронное (где phi1 != phi2)
    target_eq = None
    for eq in eqs:
        if abs(eq[0] - eq[2]) > 0.1:  # Если фазы различаются
            target_eq = eq
            break

    if target_eq is None:
        return 10.0  # Штраф, если седло не найдено

    # Вычисляем Якобиан и собственные вектора
    jac = jacobian(target_eq, gamma_val, lam, k_val)
    w, v = np.linalg.eig(jac)

    # Ищем неустойчивое направление (Re(lambda) > 0)
    unstable_idx = np.argmax(np.real(w))
    if np.real(w[unstable_idx]) < 0:
        return 10.0  # Это не седло-фокус (нет неустойчивости)

    unstable_vec = np.real(v[:, unstable_idx])

    # Стартуем чуть-чуть в стороне от седла
    start_point = np.array(target_eq) + 1e-6 * unstable_vec

    # Интегрируем
    sol = solve_ivp(system, [0, 500], start_point, args=(gamma_val, lam, k_val),
                    rtol=1e-9, atol=1e-9, max_step=0.1)

    # Возвращаем расстояние до седла
    return distance_to_saddle(sol, target_eq)


# === ЗАПУСК ПОИСКА ===
print("🔍 Ищем параметры гомоклиники...")
res = minimize_scalar(objective_function, bounds=lam_bounds, method='bounded')

best_lam = res.x
best_dist = res.fun

print(f"\n✅ РЕЗУЛЬТАТ:")
print(f"Параметры: k={k_val}, gamma={gamma_val}")
print(f"Найденный lambda: {best_lam:.6f}")
print(f"Минимальное расстояние до седла: {best_dist:.6e}")

if best_dist < 1e-3:
    print("🎉 ГОМОКЛИНИКА НАЙДЕНА! (Траектория замкнулась)")
else:
    print("⚠️ Расстояние велико. Попробуйте изменить gamma.")

# === ВИЗУАЛИЗАЦИЯ ===
# Строим график найденной траектории
eqs = find_saddle_focus(gamma_val, best_lam, k_val)
target_eq = [eq for eq in eqs if abs(eq[0] - eq[2]) > 0.1][0]

jac = jacobian(target_eq, gamma_val, best_lam, k_val)
w, v = np.linalg.eig(jac)
unstable_idx = np.argmax(np.real(w))
unstable_vec = np.real(v[:, unstable_idx])
start_point = np.array(target_eq) + 1e-6 * unstable_vec

sol = solve_ivp(system, [0, 500], start_point, args=(gamma_val, best_lam, k_val),
                rtol=1e-9, atol=1e-9, max_step=0.1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(sol.y[0], sol.y[1], 'b-', lw=1)  # phi1 vs v1
plt.scatter([target_eq[0]], [0], c='red', s=50, marker='x')
plt.title(f"Phase Portrait (phi1, v1)\nlambda={best_lam:.4f}")
plt.xlabel("phi1")
plt.ylabel("v1")

plt.subplot(1, 2, 2)
plt.plot(sol.y[2], sol.y[3], 'r-', lw=1)  # phi2 vs v2
plt.scatter([target_eq[2]], [0], c='red', s=50, marker='x')
plt.title(f"Phase Portrait (phi2, v2)")
plt.xlabel("phi2")
plt.ylabel("v2")

plt.tight_layout()
plt.show()