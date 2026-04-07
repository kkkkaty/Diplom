import math
import numpy as np
from numba import cuda, float64, int32


TPB = 256


@cuda.jit(device=True)
def rhs_pendulum(fi1, v1, fi2, v2, gamma, lam, k, out):
    out[0] = v1
    out[1] = -lam * v1 - math.sin(fi1) + gamma + k * math.sin(fi2 - fi1)
    out[2] = v2
    out[3] = -lam * v2 - math.sin(fi2) + gamma + k * math.sin(fi1 - fi2)


@cuda.jit(device=True)
def rk4_step_device(y, gamma, lam, k, dt):
    k1 = cuda.local.array(4, dtype=float64)
    k2 = cuda.local.array(4, dtype=float64)
    k3 = cuda.local.array(4, dtype=float64)
    k4 = cuda.local.array(4, dtype=float64)
    yt = cuda.local.array(4, dtype=float64)

    rhs_pendulum(y[0], y[1], y[2], y[3], gamma, lam, k, k1)

    for i in range(4):
        yt[i] = y[i] + 0.5 * dt * k1[i]
    rhs_pendulum(yt[0], yt[1], yt[2], yt[3], gamma, lam, k, k2)

    for i in range(4):
        yt[i] = y[i] + 0.5 * dt * k2[i]
    rhs_pendulum(yt[0], yt[1], yt[2], yt[3], gamma, lam, k, k3)

    for i in range(4):
        yt[i] = y[i] + dt * k3[i]
    rhs_pendulum(yt[0], yt[1], yt[2], yt[3], gamma, lam, k, k4)

    for i in range(4):
        y[i] = y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@cuda.jit
def build_init_points_kernel(
    eq_points,
    unstable_dirs,
    branch_ids,
    valid_mask,
    params_x,
    params_y,
    def_params,
    param_x_idx,
    param_y_idx,
    offset_index,
    eps_shift,
    dt_sep,
    out_inits,
):
    idx = cuda.grid(1)

    if idx >= params_x.shape[0]:
        return

    if valid_mask[idx] == 0:
        return

    gamma = def_params[0]
    lam = def_params[1]
    k = def_params[2]

    px = params_x[idx]
    py = params_y[idx]

    if param_x_idx == 0:
        gamma = px
    elif param_x_idx == 1:
        lam = px
    elif param_x_idx == 2:
        k = px

    if param_y_idx == 0:
        gamma = py
    elif param_y_idx == 1:
        lam = py
    elif param_y_idx == 2:
        k = py

    y = cuda.local.array(4, dtype=float64)

    sign = 1.0
    if branch_ids[idx] < 0:
        sign = -1.0

    for j in range(4):
        y[j] = eq_points[idx, j] + sign * eps_shift * unstable_dirs[idx, j]

    # ВАЖНО:
    # нужна только traj[offset_index], а не вся траектория длины steps_sep
    for _ in range(offset_index):
        rk4_step_device(y, gamma, lam, k, dt_sep)

    for j in range(4):
        out_inits[idx, j] = y[j]


def build_init_points_cuda(
    eq_points,
    unstable_dirs,
    branch_ids,
    params_x,
    params_y,
    def_params,
    param_x_name,
    param_y_name,
    param_to_index,
    nones,
    offset_index=1,
    eps_shift=1e-6,
    dt_sep=1e-3,
):
    total = len(params_x)

    eq_arr = np.zeros((total, 4), dtype=np.float64)
    dir_arr = np.zeros((total, 4), dtype=np.float64)
    branch_arr = np.zeros(total, dtype=np.int32)
    valid_mask = np.ones(total, dtype=np.int32)

    for idx in range(total):
        if eq_points[idx] is None or unstable_dirs[idx] is None or branch_ids[idx] is None:
            valid_mask[idx] = 0
            continue

        eq_arr[idx, :] = np.asarray(eq_points[idx], dtype=np.float64)
        dir_arr[idx, :] = np.asarray(unstable_dirs[idx], dtype=np.float64)
        branch_arr[idx] = int(branch_ids[idx])

    for idx in np.asarray(nones, dtype=np.int32):
        if 0 <= idx < total:
            valid_mask[idx] = 0

    out_inits = np.zeros((total, 4), dtype=np.float64)

    d_eq = cuda.to_device(eq_arr)
    d_dir = cuda.to_device(dir_arr)
    d_branch = cuda.to_device(branch_arr)
    d_valid = cuda.to_device(valid_mask)
    d_params_x = cuda.to_device(np.asarray(params_x, dtype=np.float64))
    d_params_y = cuda.to_device(np.asarray(params_y, dtype=np.float64))
    d_def = cuda.to_device(np.asarray(def_params, dtype=np.float64))
    d_out = cuda.to_device(out_inits)

    blocks = (total + TPB - 1) // TPB

    build_init_points_kernel[blocks, TPB](
        d_eq,
        d_dir,
        d_branch,
        d_valid,
        d_params_x,
        d_params_y,
        d_def,
        int32(param_to_index[param_x_name]),
        int32(param_to_index[param_y_name]),
        int32(offset_index),
        float(eps_shift),
        float(dt_sep),
        d_out,
    )

    out_inits = d_out.copy_to_host()
    return out_inits.reshape(-1)