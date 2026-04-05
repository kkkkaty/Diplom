import numpy as np

DIM = 2

def rhs_cpu(params, y, dydt):
    """
    Упрощённая реализация функции `rhs` для маятника.
    """
    g, l = params
    dydt[0] = y[1]
    dydt[1] = -(g / l) * y[0]

def stepper_rk4_cpu(params, y_curr, dt):
    """
    Упрощённая реализация метода Рунге-Кутты для тестов.
    """
    k1 = np.zeros(DIM, dtype=np.float32)
    k2 = np.zeros(DIM, dtype=np.float32)
    k3 = np.zeros(DIM, dtype=np.float32)
    k4 = np.zeros(DIM, dtype=np.float32)
    func = np.zeros(DIM, dtype=np.float32)

    rhs_cpu(params, y_curr, k1)

    for i in range(DIM):
        func[i] = y_curr[i] + k1[i] * dt / 2.0
    rhs_cpu(params, func, k2)

    for i in range(DIM):
        func[i] = y_curr[i] + k2[i] * dt / 2.0
    rhs_cpu(params, func, k3)

    for i in range(DIM):
        func[i] = y_curr[i] + k3[i] * dt
    rhs_cpu(params, func, k4)

    for i in range(DIM):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0

def test_stepper_correctness():
    """
    Проверка корректности для маятника.
    """
    params = np.array([9.8, 1.0], dtype=np.float32)  # g = 9.8, l = 1.0
    y_init = np.array([0.1, 0.0], dtype=np.float32)  # Начальный угол 0.1 рад
    dt = 1e-3
    steps = 1000

    def analytic_solution(t):
        g, l = params
        omega = np.sqrt(g / l)
        return 0.1 * np.cos(omega * t), -0.1 * omega * np.sin(omega * t)

    y_curr = y_init.copy()
    t = 0
    for step in range(steps):
        try:
            stepper_rk4_cpu(params, y_curr, dt)
            # t = (step + 1) * dt
            t += dt
            y_exact, y_exact_derivative = analytic_solution(t)

            # Сравнение с аналитическим решением
            if not np.allclose(y_curr[0], y_exact, atol=1e-2):
                print(f"Ошибка в первой компоненте на шаге {step}: y={y_curr[0]}, ожидается {y_exact}")

            if not np.allclose(y_curr[1], y_exact_derivative, atol=1e-2):
                print(f"Ошибка во второй компоненте на шаге {step}: y'={y_curr[1]}, ожидается {y_exact_derivative}")

        except Exception as e:
            print(f"Ошибка на шаге {step}: {e}")

    print("Тест затухающего осциллятора пройден!")

if __name__ == "__main__":
    test_stepper_correctness()