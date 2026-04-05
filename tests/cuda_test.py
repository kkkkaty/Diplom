import numpy as np
from numba import cuda

# Определение параметров
DIM = 3  # Количество уравнений
dt = 0.01  # Шаг по времени
N = 100  # Количество шагов интегрирования

# Функция правой части системы уравнений (CPU)
def rhs(params, y):
    a, b = params
    dydt = np.zeros(DIM)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -b * y[2] - y[1] + a * y[0] - a * (y[0] ** 3)
    return dydt

# Метод Рунге-Кутты (CPU)
def stepper_rk4_cpu(params, y_curr):
    k1 = rhs(params, y_curr)
    k2 = rhs(params, y_curr + k1 * dt / 2.0)
    k3 = rhs(params, y_curr + k2 * dt / 2.0)
    k4 = rhs(params, y_curr + k3 * dt)

    y_curr += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0
    return y_curr

# Функция правой части системы уравнений (GPU)
@cuda.jit
def rhs_cuda(params, y, dydt):
    a = params[0]
    b = params[1]
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -b * y[2] - y[1] + a * y[0] - a * (y[0] ** 3)

# Метод Рунге-Кутты (GPU)
@cuda.jit
def stepper_rk4_gpu(params, y_curr):
    k1 = cuda.local.array(DIM, dtype=np.float32)
    k2 = cuda.local.array(DIM, dtype=np.float32)
    k3 = cuda.local.array(DIM, dtype=np.float32)
    k4 = cuda.local.array(DIM, dtype=np.float32)

    rhs_cuda(params, y_curr, k1)

    for i in range(DIM):
        k2[i] = y_curr[i] + k1[i] * dt / 2.0
    rhs_cuda(params, k2, k2)

    for i in range(DIM):
        k3[i] = y_curr[i] + k2[i] * dt / 2.0
    rhs_cuda(params, k3, k3)

    for i in range(DIM):
        k4[i] = y_curr[i] + k3[i] * dt
    rhs_cuda(params, k4, k4)

    for i in range(DIM):
        y_curr[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0

# Основная функция для выполнения интеграции и вывода результатов
def run_integration():
    # Параметры системы
    a = 1.0
    b = 0.5
    params_cpu = np.array([a, b], dtype=np.float32)

    # Начальные условия
    y_init_cpu = np.array([1e-8, 0.0, 0.0], dtype=np.float32)  # Начальные условия для системы

    # Перенос данных на GPU
    params_gpu = cuda.to_device(params_cpu)
    y_curr_gpu = cuda.to_device(y_init_cpu.copy())

    print("Step\tCPU Result\tGPU Result")

    for step in range(N):
        # CPU
        y_init_cpu = stepper_rk4_cpu(params_cpu, y_init_cpu)

        # GPU
        stepper_rk4_gpu[(1,), (1)](params_gpu, y_curr_gpu)

        # Копирование результата обратно с GPU
        y_curr_result_gpu = y_curr_gpu.copy_to_host()

        # Вывод результатов
        print(f"{step}\t{y_init_cpu}\t{y_curr_result_gpu}")

if __name__ == "__main__":
    run_integration()