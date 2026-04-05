import numpy as np
from numba import cuda
from src.mapping.convert import decimal_to_number_system

DIM = 3
THREADS_PER_BLOCK = 512

INFINITY = 100000

InfinityError = -0.2
KneadingDoNotEndError = -0.1


@cuda.jit
def rhs(params, y, dydt):
    """Calculates the right-hand side of the system"""
    a, b = params
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -b * y[2] - y[1] + a * y[0] - a * (y[0] ** 3)


@cuda.jit
def stepper_rk4(params, y_curr, dt):
    """Makes RK-4 step and saves the value in y_curr"""
    k1 = cuda.local.array(DIM, dtype=np.float64)
    k2 = cuda.local.array(DIM, dtype=np.float64)
    k3 = cuda.local.array(DIM, dtype=np.float64)
    k4 = cuda.local.array(DIM, dtype=np.float64)
    y_temp = cuda.local.array(DIM, dtype=np.float64)

    rhs(params, y_curr, k1)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k1[i] * dt / 2.0
    rhs(params, y_temp, k2)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k2[i] * dt / 2.0
    rhs(params, y_temp, k3)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k3[i] * dt
    rhs(params, y_temp, k4)

    for i in range(DIM):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0


@cuda.jit
def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):
    """Calculates kneadings during integration"""
    # n -- количество шагов интегрирования
    # stride -- через сколько шагов начинаем считать нидинги
    # first_derivative_curr, prev -- значения производных системы на текущем шаге и на предыдущем

    first_derivative_prev = cuda.local.array(DIM, dtype=np.float64)
    first_derivative_curr = cuda.local.array(DIM, dtype=np.float64)
    kneading_index = 0
    kneadings_weighted_sum = 0

    rhs(params, y_curr, first_derivative_prev)

    for i in range(1, n):

        for j in range(stride):
            stepper_rk4(params, y_curr, dt)

        for k in range(DIM):
            if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                return InfinityError

        rhs(params, y_curr, first_derivative_curr)
        if first_derivative_prev[0] * first_derivative_curr[0] < 0:

            if first_derivative_curr[1] < 0 and y_curr[0] > 1:
                if kneading_index >= kneadings_start:
                    # 1
                    kneadings_weighted_sum += 1 / (2.0 ** (-kneading_index + kneadings_end + 1))
                kneading_index += 1

            elif first_derivative_curr[1] > 0 and y_curr[0] < -1:
                # 0
                kneading_index += 1

        first_derivative_prev[0] = first_derivative_curr[0]

        if kneading_index > kneadings_end:
            return kneadings_weighted_sum

    return KneadingDoNotEndError


@cuda.jit
def sweep_threads(
    kneadings_weighted_sum_set_gpu,
    y_inits,
    a_start,
    a_end,
    a_count,
    b_start,
    b_end,
    b_count,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    """CUDA kernel"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    a_step = (a_end - a_start) / (a_count - 1)
    b_step = (b_end - b_start) / (b_count - 1)

    if idx < a_count * b_count:

        i = idx // a_count
        j = idx % a_count

        params = cuda.local.array(2, dtype=np.float64)
        y_init = cuda.local.array(DIM, dtype=np.float64)

        params[0] = a_start + i * a_step
        params[1] = b_start + j * b_step

        y_init[0] = y_inits[idx * DIM + 0]
        y_init[1] = y_inits[idx * DIM + 1]
        y_init[2] = y_inits[idx * DIM + 2]

        kneadings_weighted_sum_set_gpu[i * b_count + j] = integrator_rk4(y_init, params, dt, n, stride,
                                                                         kneadings_start, kneadings_end)


def sweep(
    y_inits,
    a_start,
    a_end,
    a_count,
    b_start,
    b_end,
    b_count,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    """Calls CUDA kernel and gets kneadings set back from GPU"""
    total_parameter_space_size = a_count * b_count
    kneadings_weighted_sum_set = np.zeros(sweep_size * sweep_size)
    kneadings_weighted_sum_set_gpu = cuda.device_array(total_parameter_space_size)

    y_inits_gpu = cuda.device_array(len(y_inits))
    for i in range(len(y_inits)):
        y_inits_gpu[i] = y_inits[i]

    grid_x_dimension = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    dim_grid = grid_x_dimension
    dim_block = THREADS_PER_BLOCK

    print(f"Num of blocks per grid:       {dim_grid}")
    print(f"Num of threads per block:     {dim_block}")
    print(f"Total Num of threads running: {dim_grid * dim_block}")
    print(f"Parameters aCount = {a_count}, bCount = {b_count}")

    # Call CUDA kernel
    sweep_threads[dim_grid, dim_block](  # blocks, threads
        kneadings_weighted_sum_set_gpu,
        y_inits_gpu,
        a_start,
        a_end,
        a_count,
        b_start,
        b_end,
        b_count,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end,
    )

    kneadings_weighted_sum_set_gpu.copy_to_host(kneadings_weighted_sum_set)

    return kneadings_weighted_sum_set


if __name__ == "__main__":
    dt = 0.01
    n = 30000
    stride = 1
    max_kneadings = 7
    sweep_size = 10

    a_start = 0.0
    a_end = 2.2
    b_start = 0.0
    b_end = 1.5

    y_inits = [1e-8, 0.0, 0.0] * sweep_size * sweep_size

    kneadings_weighted_sum_set = sweep(
        y_inits,
        a_start,
        a_end,
        sweep_size,
        b_start,
        b_end,
        sweep_size,
        dt,
        n,
        stride,
        0,
        max_kneadings
    )

    np.savez(
        'sweep_trese.npz',
        a_start=a_start,
        a_end=a_end,
        b_start=b_start,
        b_end=b_end,
        sweep_size=sweep_size,
        kneadings=kneadings_weighted_sum_set
    )

    print("Results:")
    for idx in range(sweep_size * sweep_size):
        i = idx // sweep_size
        j = idx % sweep_size

        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_number_system(kneading_weighted_sum, 2)

        print(f"a: {a_start + i * (a_end - a_start) / (sweep_size - 1):.6f}, "
              f"b: {b_start + j * (b_end - b_start) / (sweep_size - 1):.6f} => "
              f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")