import numpy as np
import pprint
import matplotlib.pyplot as plt
from numba import cuda
from lib.computation_template.workers_utils import register, makeFinalOutname
from src.mapping.plot_kneadings import plot_mode_map, set_random_color_map
from src.system_analysis.get_inits import build_inits_on_parameter_grid_with_shape
# Импорт registry из основного воркера
from src.computing.workers_kneadings_pendulums import registry

# =============================================================================
# CUDA-УТИЛИТЫ (ПРАВИЛЬНЫЕ ОПРЕДЕЛЕНИЯ ДЛЯ ЭТОГО ФАЙЛА)
# =============================================================================

DIM = 4
INFINITY = 1e6
EPS = 1e-12


@cuda.jit(device=True)
def rhs(params, y, dydt):
    gamma = params[0]
    lam = params[1]
    k = params[2]
    fi1, v1, fi2, v2 = y[0], y[1], y[2], y[3]

    dydt[0] = v1
    dydt[1] = gamma - lam * v1 - np.sin(fi1) + k * np.sin(fi2 - fi1)
    dydt[2] = v2
    dydt[3] = gamma - lam * v2 - np.sin(fi2) + k * np.sin(fi1 - fi2)


@cuda.jit(device=True)
def stepper_rk4(params, y_curr, dt):
    k1 = cuda.local.array(DIM, dtype=np.float64)
    k2 = cuda.local.array(DIM, dtype=np.float64)
    k3 = cuda.local.array(DIM, dtype=np.float64)
    k4 = cuda.local.array(DIM, dtype=np.float64)
    y_tmp = cuda.local.array(DIM, dtype=np.float64)

    rhs(params, y_curr, k1)
    for i in range(DIM): y_tmp[i] = y_curr[i] + 0.5 * dt * k1[i]
    rhs(params, y_tmp, k2)
    for i in range(DIM): y_tmp[i] = y_curr[i] + 0.5 * dt * k2[i]
    rhs(params, y_tmp, k3)
    for i in range(DIM): y_tmp[i] = y_curr[i] + dt * k3[i]
    rhs(params, y_tmp, k4)
    for i in range(DIM):
        y_curr[i] = y_curr[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


# =============================================================================
# ЛОГИКА КЛАССИФИКАЦИИ
# =============================================================================

def decode_attractor_type(val: float) -> str:
    if np.isclose(val, 0.0, atol=1e-5): return "BOTH_OSCILLATE"
    if np.isclose(val, 1.0, atol=1e-5): return "φ1_OSC_φ2_ROT"
    if np.isclose(val, 2.0, atol=1e-5): return "φ1_ROT_φ2_OSC"
    if np.isclose(val, 3.0, atol=1e-5): return "BOTH_ROTATE"
    if val < -0.05: return f"ERROR({val})"
    return f"UNKNOWN({val})"


@cuda.jit(device=True)
def count_full_rotations(fi_start, fi_end):
    def normalize(angle):
        while angle >= np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    fi_start_norm = normalize(fi_start)
    fi_end_norm = normalize(fi_end)
    delta = fi_end_norm - fi_start_norm
    if delta > np.pi: return -1
    if delta < -np.pi: return +1
    return 0


@cuda.jit(device=True)
def classify_pendulum_motion(fi_history, length, rotation_threshold):
    if length < 2: return 0
    total_rotations = 0
    for i in range(1, length):
        total_rotations += count_full_rotations(fi_history[i - 1], fi_history[i])
    if abs(total_rotations) >= rotation_threshold:
        return 1
    return 0


@cuda.jit(device=True)
def integrator_attractor_classifier(
        y_curr, params, dt, n, stride,
        record_interval, max_record_len,
        rotation_threshold
):
    y_prev = cuda.local.array(DIM, dtype=np.float64)
    for k in range(DIM): y_prev[k] = y_curr[k]

    fi1_hist = cuda.local.array(2048, dtype=np.float64)
    fi2_hist = cuda.local.array(2048, dtype=np.float64)
    rec_idx = 0

    for step in range(n):
        for _ in range(stride):
            stepper_rk4(params, y_curr, dt)

        for k in range(DIM):
            if abs(y_curr[k]) > INFINITY:
                return -0.1

        if step % record_interval == 0 and rec_idx < max_record_len:
            fi1_hist[rec_idx] = y_curr[0]
            fi2_hist[rec_idx] = y_curr[2]
            rec_idx += 1

        for k in range(DIM): y_prev[k] = y_curr[k]

    if rec_idx < 10:
        return -0.2

    type1 = classify_pendulum_motion(fi1_hist, rec_idx, rotation_threshold)
    type2 = classify_pendulum_motion(fi2_hist, rec_idx, rotation_threshold)

    return float(2 * type1 + type2)


# =============================================================================
# ЗАПУСК (SWEEP)
# =============================================================================

def make_sweep_attractor_classifier():
    @cuda.jit
    def sweep_threads_attractor(
            result_set, inits, nones, params_x, params_y,
            def_params, param_x_idx, param_y_idx,
            up_n, down_n, left_n, right_n,
            dt, n, stride,
            record_interval, max_record_len,
            rotation_threshold
    ):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        total = (left_n + right_n + 1) * (up_n + down_n + 1)
        if idx >= total: return

        for i in range(len(nones)):
            if idx == nones[i]:
                result_set[idx] = -0.3
                return

        y = cuda.local.array(DIM, dtype=np.float64)
        base = idx * DIM
        for k in range(DIM): y[k] = inits[base + k]

        params = cuda.local.array(3, dtype=np.float64)
        for k in range(3): params[k] = def_params[k]
        params[param_x_idx] = params_x[idx]
        params[param_y_idx] = params_y[idx]

        res = integrator_attractor_classifier(
            y, params, dt, n, stride,
            record_interval, max_record_len,
            rotation_threshold
        )
        result_set[idx] = res

    return sweep_threads_attractor


def sweep_attractor_classification(
        inits, nones, params_x, params_y,
        def_params, param_to_index,
        param_x_str, param_y_str,
        up_n, down_n, left_n, right_n,
        dt, n, stride,
        record_interval=10, max_record_len=1024, rotation_threshold=3
):
    THREADS_PER_BLOCK = 512
    total = (left_n + right_n + 1) * (up_n + down_n + 1)
    result_set = np.zeros(total, dtype=np.float64)

    inits_gpu = cuda.to_device(np.asarray(inits, dtype=np.float64))
    nones_gpu = cuda.to_device(np.asarray(nones, dtype=np.int32))
    def_params_gpu = cuda.to_device(np.asarray(def_params, dtype=np.float64))
    params_x_gpu = cuda.to_device(np.asarray(params_x, dtype=np.float64))
    params_y_gpu = cuda.to_device(np.asarray(params_y, dtype=np.float64))

    param_x_idx = param_to_index[param_x_str]
    param_y_idx = param_to_index[param_y_str]

    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    print(f"[Attractor Classifier] Blocks: {blocks}, Threads: {THREADS_PER_BLOCK}")

    # Создаём буфер на GPU
    result_set_gpu = cuda.device_array(total, dtype=np.float64)

    # Запускаем ядро, передавая устройство
    kernel = make_sweep_attractor_classifier()
    kernel[blocks, THREADS_PER_BLOCK](
        result_set_gpu,
        inits_gpu, nones_gpu,
        params_x_gpu, params_y_gpu,
        def_params_gpu,
        param_x_idx, param_y_idx,
        up_n, down_n, left_n, right_n,
        dt, n, stride,
        record_interval, max_record_len,
        rotation_threshold
    )

    # Синхронизируем и копируем результат обратно на CPU
    cuda.synchronize()
    result_set_gpu.copy_to_host(result_set)  # ← ПРАВИЛЬНЫЙ МЕТОД Numba

    return result_set


# =============================================================================
# РЕГИСТРАЦИЯ
# =============================================================================

@register(registry, "init", "attractor_classification")
def init_attractor_classification(config, timeStamp):
    from src.computing.workers_kneadings_pendulums import init_kneadings_pendulums
    return init_kneadings_pendulums(config, timeStamp)


@register(registry, "worker", "attractor_classification")
def worker_attractor_classification(config, initResult, timeStamp):
    def_params = initResult["def_params"]
    grid = config["grid"]

    left_n = int(grid["first"]["left_n"])
    right_n = int(grid["first"]["right_n"])
    up_n = int(grid["second"]["up_n"])
    down_n = int(grid["second"]["down_n"])
    param_x_name = grid["first"]["name"]
    param_y_name = grid["second"]["name"]

    clf_cfg = config.get("attractor_classification", {})
    dt = float(clf_cfg.get("dt", 0.01))
    n = int(clf_cfg.get("n", 50000))
    stride = int(clf_cfg.get("stride", 1))
    record_interval = int(clf_cfg.get("record_interval", 10))
    max_record_len = int(clf_cfg.get("max_record_len", 1024))
    rotation_threshold = int(clf_cfg.get("rotation_threshold", 3))

    inits = initResult["inits"]
    nones = initResult["nones"]
    params_x = initResult["params_x"]
    params_y = initResult["params_y"]

    print(f"[Worker] Grid size: {len(inits) // DIM} points")

    # Важно: используем PARAM_TO_INDEX из sweep_pendulums, если он доступен, или хардкодим
    from src.cuda_sweep.sweep_pendulums import PARAM_TO_INDEX

    result_set = sweep_attractor_classification(
        inits=inits, nones=nones,
        params_x=params_x, params_y=params_y,
        def_params=def_params,
        param_to_index=PARAM_TO_INDEX,
        param_x_str=param_x_name, param_y_str=param_y_name,
        up_n=up_n, down_n=down_n, left_n=left_n, right_n=right_n,
        dt=dt, n=n, stride=stride,
        record_interval=record_interval,
        max_record_len=max_record_len,
        rotation_threshold=rotation_threshold
    )

    records = ""
    total = (left_n + right_n + 1) * (up_n + down_n + 1)
    for idx in range(min(20, total)):
        val = result_set[idx]
        line = (
            f"{param_x_name}: {params_x[idx]:.6f}, "
            f"{param_y_name}: {params_y[idx]:.6f} => "
            f"{decode_attractor_type(val)} (raw: {val})"
        )
        print(line)
        records += line + "\n"

    return {"result_set": result_set, "records": records}


@register(registry, "post", "attractor_classification")
def post_attractor_classification(config, initResult, workerResult, grid, startTime):
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    grid_dict = config["grid"]
    param_x_caption = grid_dict["first"]["caption"]
    param_y_caption = grid_dict["second"]["caption"]

    left_n = int(grid_dict["first"]["left_n"])
    right_n = int(grid_dict["first"]["right_n"])
    up_n = int(grid_dict["second"]["up_n"])
    down_n = int(grid_dict["second"]["down_n"])

    param_x_count = left_n + right_n + 1
    param_y_count = up_n + down_n + 1

    start_vals = {"gamma": gamma, "lambda": lam, "k": k}
    param_x_name = grid_dict["first"]["name"]
    param_y_name = grid_dict["second"]["name"]

    param_x_start = start_vals[param_x_name] - left_n * float(grid_dict["first"]["left_step"])
    param_x_end = start_vals[param_x_name] + right_n * float(grid_dict["first"]["right_step"])
    param_y_start = start_vals[param_y_name] - down_n * float(grid_dict["second"]["down_step"])
    param_y_end = start_vals[param_y_name] + up_n * float(grid_dict["second"]["up_step"])

    raw_data = workerResult["result_set"]
    plot_data = np.full_like(raw_data, -1, dtype=np.float64)

    mask_both_osc = np.isclose(raw_data, 0.0, atol=1e-5)
    mask_osc_rot = np.isclose(raw_data, 1.0, atol=1e-5)
    mask_rot_osc = np.isclose(raw_data, 2.0, atol=1e-5)
    mask_both_rot = np.isclose(raw_data, 3.0, atol=1e-5)
    mask_error = raw_data < -0.05

    plot_data[mask_both_osc] = 0
    plot_data[mask_osc_rot] = 1
    plot_data[mask_rot_osc] = 2
    plot_data[mask_both_rot] = 3
    plot_data[mask_error] = -1

    from matplotlib.colors import ListedColormap
    four_color_cmap = ListedColormap([
        '#4DAF4A', '#377EB8', '#E41A1C', '#984EA3', '#999999'
    ])

    plt.figure(figsize=(10, 8))
    plot_mode_map(
        plot_data, lambda: four_color_cmap,
        param_x_caption, param_y_caption,
        param_x_start, param_x_end, param_x_count,
        param_y_start, param_y_end, param_y_count,
        font_size=14
    )
    plt.clim(-0.5, 3.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4DAF4A', label='Both oscillate'),
        Patch(facecolor='#377EB8', label='φ1 osc, φ2 rot'),
        Patch(facecolor='#E41A1C', label='φ1 rot, φ2 osc'),
        Patch(facecolor='#984EA3', label='Both rotate'),
        Patch(facecolor='#999999', label='Error')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.title(f"Attractor Types: γ={gamma}, λ={lam}, k={k}", fontsize=14)

    npy_outname = makeFinalOutname(config, initResult, "npy", startTime)
    np.save(npy_outname, raw_data)

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, "w", encoding="utf-8") as f:
        f.write(workerResult["records"])

    img_ext = config["output"]["imageExtension"]
    plot_outname = makeFinalOutname(config, initResult, img_ext, startTime)
    plt.savefig(plot_outname, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[Post] Map saved to {plot_outname}")
    return {"plot_saved": plot_outname}