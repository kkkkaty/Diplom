import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# === ПАРАМЕТРЫ ===
k_val = 0.06
gamma_val = 0.95
lam_bounds = (0.01, 0.5)


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


def find_saddle_foci(gamma, lam, k):
    from scipy.optimize import fsolve
    def equations(x):
        p1, p2 = x
        return [
            gamma - np.sin(p1) + k * np.sin(p2 - p1),
            gamma - np.sin(p2) + k * np.sin(p1 - p2)
        ]

    guesses = [(1.5, 2.0), (2.0, 1.5), (0.5, 0.5)]
    equilibria = []
    for g in guesses:
        sol = fsolve(equations, g)
        is_new = True
        for eq in equilibria:
            if np.allclose(sol, eq[:2], atol=1e-5):
                is_new = False
                break
        if is_new:
            equilibria.append([sol[0], 0.0, sol[1], 0.0])
    return equilibria


def compute_min_distance(lam):
    eqs = find_saddle_foci(gamma_val, lam, k_val)
    if len(eqs) < 2:
        return 1e6

    O1, O2 = eqs[0], eqs[1]

    # Проверка на симметрию (важно для гетероклиники)
    if not np.allclose([O1[0], O1[2]], [O2[2], O2[0]], atol=1e-4):
        return 1e6

    # Собственный вектор неустойчивого направления O1
    jac1 = jacobian(O1, gamma_val, lam, k_val)
    w, v = np.linalg.eig(jac1)
    unstable_idx = np.argmax(np.real(w))
    unstable_vec = np.real(v[:, unstable_idx])

    # Старт сепаратрисы
    start = np.array(O1) + 1e-7 * unstable_vec

    # Интегрирование
    sol = solve_ivp(system, [0, 300], start, args=(gamma_val, lam, k_val),
                    rtol=1e-9, atol=1e-9, max_step=0.05, dense_output=False)

    # Минимальное расстояние до O2 по всей траектории
    dists = np.linalg.norm(sol.y.T - O2, axis=1)
    return np.min(dists)


# === ЗАПУСК ПОИСКА ===
print("🔍 Ищу оптимальный lambda...")

# Сначала найдём и выведем информацию о седловых точках для среднего lambda
test_lam = (lam_bounds[0] + lam_bounds[1]) / 2
eqs = find_saddle_foci(gamma_val, test_lam, k_val)

print(f"\n=== ИНФОРМАЦИЯ О СЕДЛОВЫХ ТОЧКАХ ===")
print(f"Параметры: k={k_val}, gamma={gamma_val}, lambda≈{test_lam:.4f}")
print(f"Найдено состояний равновесия: {len(eqs)}")

if len(eqs) >= 2:
    O1, O2 = eqs[0], eqs[1]

    print(f"\nO1 (седло-фокус 1):")
    print(f"  Координаты: φ1={O1[0]:.8f}, v1={O1[1]:.8f}, φ2={O1[2]:.8f}, v2={O1[3]:.8f}")

    # Собственные числа для O1
    jac1 = jacobian(O1, gamma_val, test_lam, k_val)
    eig1 = np.linalg.eigvals(jac1)
    print(f"  Собственные числа: {eig1}")
    print(f"  Тип: {'седло-фокус' if np.any(np.imag(eig1) != 0) and np.any(np.real(eig1) > 0) else 'другой'}")

    print(f"\nO2 (седло-фокус 2):")
    print(f"  Координаты: φ1={O2[0]:.8f}, v1={O2[1]:.8f}, φ2={O2[2]:.8f}, v2={O2[3]:.8f}")

    # Собственные числа для O2
    jac2 = jacobian(O2, gamma_val, test_lam, k_val)
    eig2 = np.linalg.eigvals(jac2)
    print(f"  Собственные числа: {eig2}")
    print(f"  Тип: {'седло-фокус' if np.any(np.imag(eig2) != 0) and np.any(np.real(eig2) > 0) else 'другой'}")

    # Проверка симметрии
    is_symmetric = np.allclose([O1[0], O1[2]], [O2[2], O2[0]], atol=1e-4)
    print(f"\nПроверка симметрии O1 ↔ O2:")
    print(f"  O1[φ1, φ2] = [{O1[0]:.6f}, {O1[2]:.6f}]")
    print(f"  O2[φ2, φ1] = [{O2[2]:.6f}, {O2[0]:.6f}]")
    print(f"  Симметричны: {'✓ ДА' if is_symmetric else '✗ НЕТ'}")

print(f"\n{'=' * 50}")
print("🔍 Ищу оптимальный lambda методом минимизации...")

res = minimize_scalar(compute_min_distance, bounds=lam_bounds, method='bounded',
                      options={'xatol': 1e-7, 'maxiter': 50})

best_lam = res.x
best_dist = res.fun

print(f"\n{'=' * 50}")
print("✅ РЕЗУЛЬТАТ ПОИСКА ГЕТЕРОКЛИНИКИ:")
print(f"{'=' * 50}")
print(f"Оптимальный lambda: {best_lam:.8f}")
print(f"Минимальное расстояние до O2: {best_dist:.6e}")

if best_dist < 1e-3:
    print(f"\n🎉 ПОПАДАНИЕ! Это гетероклиническая траектория.")
    print(f"   Точность: {best_dist:.2e}")
else:
    print(f"\n⚠️ Расстояние всё ещё велико.")
    print(f"   Попробуйте сместить gamma на ±0.001 и повторить.")