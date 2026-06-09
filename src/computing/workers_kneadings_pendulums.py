import numpy as np
import pprint
import matplotlib.pyplot as plt

from lib.computation_template.workers_utils import register, makeFinalOutname
from src.cuda_sweep.sweep_pendulums import sweep, PARAM_TO_INDEX
from src.mapping.plot_kneadings import plot_mode_map, set_random_color_map
from src.system_analysis.get_inits import build_inits_on_parameter_grid_with_shape

registry = {
    "worker": {}, #основные вычислительные функции
    "init": {}, #функции подготовки начальных данных
    "post": {} #функции постобработки результатов
}


def decode_regime_bits(val: float) -> str:
    if val == 1.0: return "REGULAR (Periodic)"
    if val == 0.0: return "CHAOS (Irregular)"
    return f"Error/Static ({val})"

def decode_attractor_full(val: float) -> str:
    if val < -0.05:  return "Апериодичный"
    mode_idx = int(val)
    # Если из-за ошибки точности или перелива индекс стал 9 или больше
    if mode_idx > 8: mode_idx = 8
    fraction_int = int(round((val - mode_idx) * 10 ** 12))
    n2_neg = (fraction_int // 1) % 1000
    n2_pos = (fraction_int // 1000) % 1000
    n1_neg = (fraction_int // 1000000) % 1000
    n1_pos = (fraction_int // 1000000000) % 1000

    types = ["Покой", "Колебания", "Вращение"]
    t1, t2 = mode_idx // 3, mode_idx % 3

    res = f"РЕЖИМ: М1={types[t1]}, М2={types[t2]} "
    res += f"(N1: +{n1_pos}/-{n1_neg}, N2: +{n2_pos}/-{n2_neg})"
    return res


# восстанавливает из float x строку цифр 0,...,7 длины length
def decode_base25_weighted(x: float, length: int) -> str:
    if x < 0:
        return str(x)  # для -0.1/-0.2/-0.3
    symbols = []
    v = float(x)
    # С основанием 25 мы можем надежно декодировать до 10 символов подряд
    safe_length = min(length, 10)
    for _ in range(safe_length):
        v *= 25.0
        combined_symbol = int(v + 1e-11)
        # Ограничиваем значение символа от 0 до 44
        combined_symbol = max(0, min(24, combined_symbol))
        # Разделяем на m1 и m2 (например, 12 -> "12", 4 -> "04")
        m1 = combined_symbol // 5
        m2 = combined_symbol % 5
        # Записываем сам индекс совместного события от 00 до 24
        symbols.append(f"{combined_symbol:02d}")
        # Оставляем только дробную часть для следующего шага
        v -= combined_symbol
    return " - ".join(symbols)


# строит двумерную сетку значений двух параметров x и y,
# но хранит её в виде двух одномерных массивов params_x и params_y
def _generate_parameters_2d(
    start_x,
    start_y,
    up_n,
    down_n,
    left_n,
    right_n,
    up_step,
    down_step,
    left_step,
    right_step,
):
    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    total = cols * rows

    params_x = np.empty(total, dtype=np.float64)
    params_y = np.empty(total, dtype=np.float64)

    for j in range(rows):
        for i in range(cols):
            idx = i + j * cols

            di = i - left_n
            dj = j - down_n

            dx = di * (right_step if di > 0 else left_step)
            dy = dj * (up_step if dj > 0 else down_step)

            params_x[idx] = start_x + dx
            params_y[idx] = start_y + dy

    return params_x, params_y


#Проверяет, что базовые параметры системы положительны.
def _validate_positive_default_system(gamma: float, lam: float, k: float) -> None:
    bad = []
    if gamma <= 0:
        bad.append(f"gamma={gamma}")
    if lam <= 0:
        bad.append(f"lambda={lam}")
    if k <= 0:
        bad.append(f"k={k}")
    if bad:
        raise ValueError(
            "Параметры системы должны быть строго положительными: "
            + ", ".join(bad)
        )


#Помечает как пропущенные все точки сетки, где хотя бы один из параметров gamma, lambda, k <= 0.
def _append_nonpositive_param_points(
    nones: np.ndarray,
    params_x: np.ndarray,
    params_y: np.ndarray,
    param_x_name: str,
    param_y_name: str,
    gamma: float,
    lam: float,
    k: float,
) -> np.ndarray:
    total = len(params_x)
    invalid_idx = []
    for idx in range(total):
        curr_gamma = gamma
        curr_lam = lam
        curr_k = k
        if param_x_name == "gamma":
            curr_gamma = params_x[idx]
        elif param_x_name == "lambda":
            curr_lam = params_x[idx]
        elif param_x_name == "k":
            curr_k = params_x[idx]
        if param_y_name == "gamma":
            curr_gamma = params_y[idx]
        elif param_y_name == "lambda":
            curr_lam = params_y[idx]
        elif param_y_name == "k":
            curr_k = params_y[idx]
        if curr_gamma <= 0 or curr_lam <= 0 or curr_k <= 0:
            invalid_idx.append(idx)
    if not invalid_idx:
        return nones
    invalid_idx = np.asarray(invalid_idx, dtype=np.int32)
    if nones.size == 0:
        return invalid_idx
    return np.unique(np.concatenate([nones, invalid_idx])).astype(np.int32)


@register(registry, "init", "kneadings_pendulums")
def init_kneadings_pendulums(config, timeStamp):
    """
    Создаём:
      - inits: 4D начальные условия для каждой точки сетки
      - params_x/params_y: значения параметров по сетке
      - nones: индексы точек, где старт не построен или параметры запрещены
    """
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    _validate_positive_default_system(gamma, lam, k)

    grid = config["grid"]

    up_n = int(grid["second"]["up_n"])
    up_step = float(grid["second"]["up_step"])
    down_n = int(grid["second"]["down_n"])
    down_step = float(grid["second"]["down_step"])

    left_n = int(grid["first"]["left_n"])
    left_step = float(grid["first"]["left_step"])
    right_n = int(grid["first"]["right_n"])
    right_step = float(grid["first"]["right_step"])

    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    total = cols * rows

    init_mode = config.get("init_mode", "manual")
    eq_points = None

    param_x_name = grid["first"]["name"]
    param_y_name = grid["second"]["name"]

    start_vals = {"gamma": gamma, "lambda": lam, "k": k}

    if param_x_name not in start_vals or param_y_name not in start_vals:
        raise KeyError(
            f"grid.first.name / grid.second.name must be in {list(start_vals.keys())}, "
            f"got: {param_x_name}, {param_y_name}"
        )

    params_x, params_y = _generate_parameters_2d(
        start_vals[param_x_name],
        start_vals[param_y_name],
        up_n,
        down_n,
        left_n,
        right_n,
        up_step,
        down_step,
        left_step,
        right_step,
    )

    # заранее исключаем точки с неположительными параметрами
    nones = np.array([], dtype=np.int32)
    nones = _append_nonpositive_param_points(
        nones=nones,
        params_x=params_x,
        params_y=params_y,
        param_x_name=param_x_name,
        param_y_name=param_y_name,
        gamma=gamma,
        lam=lam,
        k=k,
    )

    if init_mode == "manual":
        init_cfg = config["inits"]

        fi1_0 = float(init_cfg["fi1"])
        v1_0 = float(init_cfg["v1"])
        fi2_0 = float(init_cfg["fi2"])
        v2_0 = float(init_cfg["v2"])

        y0 = np.array([fi1_0, v1_0, fi2_0, v2_0], dtype=np.float64)
        inits = np.array(y0.tolist() * total, dtype=np.float64)

    elif init_mode == "separatrix":
        sep_cfg = config["separatrix_init"]
        saddle_focus_rule = sep_cfg.get("saddle_focus_rule", "phi1_lt_phi2")
        branch_rule = sep_cfg.get("branch_rule", "phi1_above_eq")
        offset_index = int(sep_cfg.get("offset_index", 1))

        def_params = np.array([gamma, lam, k], dtype=np.float64)

        inits, sep_nones, eq_points = build_inits_on_parameter_grid_with_shape(
            params_x=params_x,
            params_y=params_y,
            def_params=def_params,
            param_x_name=param_x_name,
            param_y_name=param_y_name,
            param_to_index=PARAM_TO_INDEX,
            cols=cols,
            rows=rows,
            center_i=left_n,
            center_j=down_n,
            saddle_focus_rule=saddle_focus_rule,
            branch_rule=branch_rule,
            offset_index=offset_index,
            eps_shift=1e-6,
            dt_sep=1e-3,
            steps_sep=1,
        )

        if nones.size == 0:
            nones = sep_nones
        elif sep_nones.size != 0:
            nones = np.unique(np.concatenate([nones, sep_nones])).astype(np.int32)

        print(f"INIT GRID FROM SEPARATRIX was built, failed points: {len(nones)}")

        if eq_points is not None:
            print("FIRST 5 EQ POINTS:")
            shown = 0
            for i, eq in enumerate(eq_points):
                if eq is not None:
                    print(i, eq)
                    shown += 1
                if shown == 5:
                    break

     # ДОБАВЛЯЕМ НОВЫЙ БЛОК:
    elif init_mode == "stable_equilibrium":
        from src.system_analysis.get_inits import build_stable_inits_on_parameter_grid_with_shape
        def_params = np.array([gamma, lam, k], dtype=np.float64)

        inits, sep_nones, eq_points = build_stable_inits_on_parameter_grid_with_shape(
            params_x=params_x,
            params_y=params_y,
            def_params=def_params,
            param_x_name=param_x_name,
            param_y_name=param_y_name,
            param_to_index=PARAM_TO_INDEX,
            cols=cols,
            rows=rows,
            center_i=left_n,
            center_j=down_n,
            eps_shift=1e-4
        )
        if nones.size == 0:
            nones = sep_nones
        elif sep_nones.size != 0:
            nones = np.unique(np.concatenate([nones, sep_nones])).astype(np.int32)

        print(f"INIT GRID FROM STABLE EQUILIBRIUM was built, failed points: {len(nones)}")

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    print("FIRST 5 INIT POINTS:")
    for i in range(min(5, total)):
        print(i, inits[4 * i:4 * i + 4])

    return {
        "inits": inits,
        "nones": nones,
        "params_x": params_x,
        "params_y": params_y,
        "targetDir": config["output"]["directory"],
        "def_params": np.array([gamma, lam, k], dtype=np.float64),
        "eq_points": eq_points,
    }


@register(registry, "worker", "kneadings_pendulums")
def worker_kneadings_pendulums(config, initResult, timeStamp):
    def_params = initResult["def_params"]

    print("INIT GRID SIZE:", len(initResult["inits"]) // 4)
    print("NUM FAILED INIT POINTS:", len(initResult["nones"]))

    grid = config["grid"]
    left_n = int(grid["first"]["left_n"])
    right_n = int(grid["first"]["right_n"])
    up_n = int(grid["second"]["up_n"])
    down_n = int(grid["second"]["down_n"])
    param_x_name = grid["first"]["name"]
    param_y_name = grid["second"]["name"]

    knead_cfg = config["kneadings_pendulums"]
    use_regime_mode = knead_cfg.get("use_regime_mode", False)
    use_9color_mode = knead_cfg.get("use_9color_mode", False)
    use_continuation_mode = knead_cfg.get("use_continuation_mode", False)

    dt = float(knead_cfg["dt"])
    n = int(knead_cfg["n"])
    stride = int(knead_cfg["stride"])
    kneadings_start = int(knead_cfg["kneadings_start"])
    kneadings_end = int(knead_cfg["kneadings_end"])

    inits = initResult["inits"]
    nones = initResult["nones"]
    params_x = initResult["params_x"]
    params_y = initResult["params_y"]

    kneadings_records = pprint.pformat(config) + "\n\n"
    kneadings_records += f"INIT GRID SIZE: {len(inits) // 4}\n"
    kneadings_records += f"NUM FAILED INIT POINTS: {len(nones)}\n\n"

    kneadings_weighted_sum_set = sweep(
        inits=inits,
        nones=nones,
        params_x=params_x,
        params_y=params_y,
        def_params=def_params,
        param_to_index=PARAM_TO_INDEX,
        param_x_str=param_x_name,
        param_y_str=param_y_name,
        up_n=up_n,
        down_n=down_n,
        left_n=left_n,
        right_n=right_n,
        dt=dt,
        n=n,
        stride=stride,
        kneadings_start=kneadings_start,
        kneadings_end=kneadings_end,
        use_regime_mode=use_regime_mode,
        use_9color_mode=use_9color_mode,
        use_continuation_mode=use_continuation_mode,
    )

    total = (left_n + right_n + 1) * (up_n + down_n + 1)
    seq_len = kneadings_end - kneadings_start + 1

    for idx in range(total):
        val = float(kneadings_weighted_sum_set[idx])

        if use_9color_mode:
            knead_sym = decode_attractor_full(val)
        elif use_regime_mode:
            knead_sym = decode_regime_bits(val)
        else:
            knead_sym = decode_base25_weighted(val, seq_len)
        line = (
            f"{param_x_name}: {params_x[idx]:.15f}, "
            f"{param_y_name}: {params_y[idx]:.15f} => "
            f"{knead_sym}"
        )

        print(line)
        kneadings_records += f"{line} (Raw: {val})\n"

    return {
        "kneadings_weighted_sum_set": kneadings_weighted_sum_set,
        "kneadings_records": kneadings_records,
    }



@register(registry, "post", "kneadings_pendulums")
def post_kneadings_pendulums(config, initResult, workerResult, grid, startTime):
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    plot_params = config.get("misc", {}).get("plot_params", {"font_size": 12})
    font_size = int(plot_params.get("font_size", 12))

    grid_dict = config["grid"]
    up_n = int(grid_dict["second"]["up_n"])
    up_step = float(grid_dict["second"]["up_step"])
    down_n = int(grid_dict["second"]["down_n"])
    down_step = float(grid_dict["second"]["down_step"])
    left_n = int(grid_dict["first"]["left_n"])
    left_step = float(grid_dict["first"]["left_step"])
    right_n = int(grid_dict["first"]["right_n"])
    right_step = float(grid_dict["first"]["right_step"])

    kneadings_weighted_sum_set = workerResult["kneadings_weighted_sum_set"]
    kneadings_records = workerResult["kneadings_records"]

    param_x_caption = f"{grid_dict['first']['caption']}"
    param_y_caption = f"{grid_dict['second']['caption']}"

    param_x_count = left_n + right_n + 1
    param_y_count = up_n + down_n + 1

    start_vals = {"gamma": gamma, "lambda": lam, "k": k}
    param_x_name = grid_dict["first"]["name"]
    param_y_name = grid_dict["second"]["name"]

    param_x_start = start_vals[param_x_name] - left_n * left_step
    param_x_end = start_vals[param_x_name] + right_n * right_step
    param_y_start = start_vals[param_y_name] - down_n * down_step
    param_y_end = start_vals[param_y_name] + up_n * up_step

    raw_data = workerResult["kneadings_weighted_sum_set"]
    knead_cfg = config["kneadings_pendulums"]
    use_regime_mode = knead_cfg.get("use_regime_mode", False)
    use_9color_mode = knead_cfg.get("use_9color_mode", False)

    if use_9color_mode:
        plot_data = np.zeros_like(raw_data)
        plot_data[raw_data < -0.5] = 9  # Черный (Хаос)
        mode_indices = np.floor(raw_data + 1e-10).astype(int)
        for i in range(9):
            mask = (raw_data >= 0) & (mode_indices == i)
            plot_data[mask] = i

        from matplotlib.colors import ListedColormap
        # Контрастная палитра
        colors = [
            '#FFFFFF', # 0: S-S
            '#FFFF00', # 1: S-L
            '#FF6600', # 2: S-R
            '#00FF00', # 3: L-S
            '#006400', # 4: L-L
            '#00FFFF', # 5: L-R
            '#0000FF', # 6: R-S (Синий)
            '#9b59b6', # 7: R-L (Фиолетовый)
            '#FF00FF', # 8: R-R (Розовый)
            '#000000', # 9: Хаос (Черный)
        ]
        cmap_to_use = lambda: ListedColormap(colors)

    elif use_regime_mode:
        plot_data = np.full_like(raw_data, 2)
        plot_data[np.isclose(raw_data, 0.0, atol=1e-5)] = 1
        plot_data[raw_data < -0.2] = 0
        from matplotlib.colors import ListedColormap
        cmap_to_use = lambda: ListedColormap(['red', 'black', 'white'])
    else:
        # unique_vals — массив уникальных значений из raw_data, отсортированных по возрастанию
        #inverse_indices — массив той же длины, что и raw_data, где каждому элементу соответствует индекс его уникального значения
        unique_vals, inverse_indices = np.unique(raw_data, return_inverse=True)
        N_unique = len(unique_vals)

        # нормализуем индексы на отрезок [0, 1] (потому что в plot_mode_map vmin=0, vmax=1)
        if N_unique > 1:
            plot_data = inverse_indices / (N_unique - 1)
        else:
            plot_data = np.zeros_like(raw_data)

        # передаем в качестве палитры функцию генерации случайных цветов
        cmap_to_use = set_random_color_map

    # Отрисовка
    plt.figure(figsize=(14, 9))  # Широкое окно для легенды
    plot_mode_map(
        plot_data, cmap_to_use,
        param_x_caption, param_y_caption,
        param_x_start, param_x_end, param_x_count,
        param_y_start, param_y_end, param_y_count,
        font_size,
    )

    # Настройка легенды (Русский язык)
    if use_9color_mode:
        plt.clim(-0.5, 9.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', label='М1:Покой, М2:Покой'),
            Patch(facecolor='#FFFF00', label='М1:Покой, М2:Колебания'),
            Patch(facecolor='#FF6600', label='М1:Покой, М2:Вращение'),
            Patch(facecolor='#00FF00', label='М1:Колебания, М2:Покой'),
            Patch(facecolor='#006400', label='М1:Колебания, М2:Колебания'),
            Patch(facecolor='#00FFFF', label='М1:Колебания, М2:Вращение'),
            Patch(facecolor='#0000FF', label='М1:Вращение, М2:Покой'),
            Patch(facecolor='#9b59b6', label='М1:Вращение, М2:Колебания'),
            Patch(facecolor='#FF00FF', label='М1:Вращение, М2:Вращение'),
            Patch(facecolor='#000000', label='Апериодичный режим (Хаос)'),
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                   title="Типы аттракторов", fontsize=font_size - 2)
    elif use_regime_mode:
        plt.clim(0, 2)

    plt.title(fr"$\gamma={gamma}$, $\lambda={lam}$, $k={k}$", fontsize=font_size)

    npy_outname = makeFinalOutname(config, initResult, "npy", startTime)
    np.save(npy_outname, kneadings_weighted_sum_set)

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, "w", encoding="utf-8") as f:
        f.write(kneadings_records)

    img_extension = config["output"]["imageExtension"]
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    plt.savefig(plot_outname, dpi=600, bbox_inches="tight")
    plt.close()