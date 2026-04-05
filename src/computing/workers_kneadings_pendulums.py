import numpy as np
import pprint
import matplotlib.pyplot as plt

from lib.computation_template.workers_utils import register, makeFinalOutname
from src.cuda_sweep.sweep_pendulums import sweep, PARAM_TO_INDEX
from src.mapping.plot_kneadings import plot_mode_map, set_random_color_map

registry = {
    "worker": {},
    "init": {},
    "post": {}
}

#восстанавливает из float x строку цифр 0,...,7 длины length
def decode_base8_weighted(x: float, length: int) -> str:
    """
    x = sum(d_i / 8^(length-i)), d_i in {0..7}
    Возвращает строку из length цифр 0..7.
    """
    if x < 0:
        return str(x)  # для -0.1/-0.2/-0.3

    s = []
    v = float(x)

    for _ in range(length):
        v *= 8.0 #умножили на 8, целая часть стала следующей цифрой. остаток дроби оставляем на след шаг
        d = int(v + 1e-12) #1e-12 слегка подталкивает число вверх, снижая риск ошибок
        #защита от погрешностей
        if d < 0:
            d = 0
        if d > 7:
            d = 7
        s.append(str(d)) #добавляем цифру как символ
        v -= d #удаляем целую часть, оставляем дробную
    return "".join(s) #склеиваем список символов в строку

#строит двумерную сетку значений двух параметров x и y, но хранит её в виде двух одномерных массивов params_x и params_y
def _generate_parameters_2d(
    start_x, #центральные значения параметров, вокруг которых строится сетка
    start_y,
    up_n,  #сколько точек вверх и вниз от центра
    down_n,
    left_n, #сколько точек влево и вправо от центра
    right_n,
    up_step, #размер шага
    down_step,
    left_step,
    right_step,
):
    """
    Генерация массивов params_x/params_y в том же порядке индексов,
    как в остальном проекте: index = i + j*(left_n+right_n+1)
    """
    cols = left_n + right_n + 1 #число столбцов по x (+1 так как есть центральная точка)
    rows = up_n + down_n + 1 #число рядов по y (+1 так как есть центральная точка)
    total = cols * rows #общее число точек сетки

    #создание массивов нужного размера. тут мусор, но так быстрее, чем с нулями zeros()
    params_x = np.empty(total, dtype=np.float64)
    params_y = np.empty(total, dtype=np.float64)

    for j in range(rows):
        for i in range(cols):
            idx = i + j * cols #массив устроен как строки подряд  (сначала идут все i для j=0, потом все i для j=1 и тд)

            di = i - left_n #сколько шагов влево или вправо от центра
            dj = j - down_n #сколько шагов вниз или вверх от центра
            # смещение относительно центра (left_n, down_n)
            dx = di * (right_step if di > 0 else left_step) #если di больше нуля, делаем шаги право, иначе влево
            dy = dj * (up_step if dj > 0 else down_step)

            #реальные координаты точек в пространстве параметров
            params_x[idx] = start_x + dx
            params_y[idx] = start_y + dy

    return params_x, params_y


#инит стадия для моей задачи (подготовительная)
@register(registry, "init", "kneadings_pendulums")
def init_kneadings_pendulums(config, timeStamp):
    """
    создаём:
      - inits: 4D начальные условия для каждой точки сетки
      - params_x/params_y: значения параметров по сетке
    """

    #извлечение параметров системы по умолчанию из конфига
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    # извлечение параметров сетки из конфига
    grid = config["grid"]
    up_n = int(grid["second"]["up_n"])
    up_step = float(grid["second"]["up_step"])
    down_n = int(grid["second"]["down_n"])
    down_step = float(grid["second"]["down_step"])

    left_n = int(grid["first"]["left_n"])
    left_step = float(grid["first"]["left_step"])
    right_n = int(grid["first"]["right_n"])
    right_step = float(grid["first"]["right_step"])

    #вычисление размеров сетки
    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    total = cols * rows

    #проверка наличия начальных условий в конфиге
    if "inits" not in config:
        raise KeyError("Config must contain 'inits' section with fi1, v1, fi2, v2")

    init_cfg = config["inits"] #получаем словарь с начальными условиями
    required_keys = ("fi1", "v1", "fi2", "v2") #определяем обязательные ключи
    #для каждого обязательного ключа определяем, есть ли он в словаре
    for key in required_keys:
        if key not in init_cfg:
            raise KeyError(f"Missing initial condition '{key}' in config['inits']")

    #извлечение начальных условий
    fi1_0 = float(init_cfg["fi1"])
    v1_0 = float(init_cfg["v1"])
    fi2_0 = float(init_cfg["fi2"])
    v2_0 = float(init_cfg["v2"])

    # создание массива начальных условий [fi1, v1, fi2, v2] для каждой точки сетки
    """
    как это выглядит в памяти?
    inits = [
        fi1_0, v1_0, fi2_0, v2_0,  # точка 1
        fi1_0, v1_0, fi2_0, v2_0,  # точка 2
        fi1_0, v1_0, fi2_0, v2_0,  # точка 3
        ...  # и так для всех 361,201 точек
    ]
    """
    inits = np.array([fi1_0, v1_0, fi2_0, v2_0] * total, dtype=np.float64)

    nones = np.array([], dtype=np.int32) #запасной массив для пропущенных точек

    #определение имен параметров
    param_x_name = grid["first"]["name"]
    param_y_name = grid["second"]["name"]

    #словарь начальных значений параметров
    start_vals = {"gamma": gamma, "lambda": lam, "k": k}

    #проверяет, есть ли 1 и 2 параметры в списке допустимых
    if param_x_name not in start_vals or param_y_name not in start_vals:
        raise KeyError(
            f"grid.first.name / grid.second.name must be in {list(start_vals.keys())}, "
            f"got: {param_x_name}, {param_y_name}"
        )

    #генерация сетки параметров
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

    return {
        "inits": inits, #начальные условия для всех точек
        "nones": nones, #пустой массив
        "params_x": params_x, #значения первого параметра
        "params_y": params_y, #значения первого параметра
        "targetDir": config["output"]["directory"], #куда сохранять результаты (берет из конфига)
    }


#воркер стадия для моей задачи (главный вычислительный этап, берет уже подготовленные начальные условия)
@register(registry, "worker", "kneadings_pendulums")
def worker_kneadings_pendulums(config, initResult, timeStamp):
    # извлечение параметров системы по умолчанию из конфига
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])
    #создает массив фиксированных параметров
    def_params = np.array([gamma, lam, k], dtype=np.float64)

    #извлечение параметров сетки
    grid = config["grid"]
    left_n = int(grid["first"]["left_n"])
    right_n = int(grid["first"]["right_n"])
    up_n = int(grid["second"]["up_n"])
    down_n = int(grid["second"]["down_n"])
    param_x_name = grid["first"]["name"] #имена сканируемых параметров
    param_y_name = grid["second"]["name"]

    #извелечение параметров интегрирования
    knead_cfg = config["kneadings_pendulums"]
    dt = float(knead_cfg["dt"])
    n = int(knead_cfg["n"])
    stride = int(knead_cfg["stride"])
    kneadings_start = int(knead_cfg["kneadings_start"])
    kneadings_end = int(knead_cfg["kneadings_end"])

    #получение данных из инициализации
    inits = initResult["inits"] #начальные условия для всех точек
    nones = initResult["nones"] #пустой массив
    params_x = initResult["params_x"] #значения первого параметра
    params_y = initResult["params_y"] #значения второго параметра

    #форматирует весь конфиг в строку для текстового файла
    kneadings_records = pprint.pformat(config) + "\n\n"

    #CUDA вычисления
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
    )

    total = (left_n + right_n + 1) * (up_n + down_n + 1) #количество точек в сетке
    seq_len = kneadings_end - kneadings_start + 1 #сколько символов в нидинге нужно восстановить

    #обработка результатов
    for idx in range(total):
        val = float(kneadings_weighted_sum_set[idx]) # получаем "упакованное" число val
        knead_sym = decode_base8_weighted(val, seq_len) #число val в виде строки 0123 длины seq_len
        #формируется строка с результатами. line = "gamma: 0.970000000000000, k: 0.060000000000000 => 01234567"
        line = (
            f"{param_x_name}: {params_x[idx]:.15f}, "
            f"{param_y_name}: {params_y[idx]:.15f} => "
            f"{knead_sym}"
        )

        # печать в консоль
        print(line)

        # запись в файл txt
        kneadings_records += f"{line} (Raw: {val})\n"

    #возвращает числовой массив (для дальнейшей обработки) и текстовые записи (для сохранения в файл)
    return {
        "kneadings_weighted_sum_set": kneadings_weighted_sum_set,
        "kneadings_records": kneadings_records,
    }


#этап пост обработки для моей задачи (берет то, что уже посчитано в воркере)
@register(registry, "post", "kneadings_pendulums")
def post_kneadings_pendulums(config, initResult, workerResult, grid, startTime):
    #достает фиксированные параметры для заголовков графиков
    def_sys = config["defaultSystem"]
    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    #настройка визуализации
    plot_params = config.get("misc", {}).get("plot_params", {"font_size": 12})
    font_size = int(plot_params.get("font_size", 12))

    #извлечение параметров сетки для построения осей графиков
    grid_dict = config["grid"]
    up_n = int(grid_dict["second"]["up_n"])
    up_step = float(grid_dict["second"]["up_step"])
    down_n = int(grid_dict["second"]["down_n"])
    down_step = float(grid_dict["second"]["down_step"])
    left_n = int(grid_dict["first"]["left_n"])
    left_step = float(grid_dict["first"]["left_step"])
    right_n = int(grid_dict["first"]["right_n"])
    right_step = float(grid_dict["first"]["right_step"])

    #получение результатов
    kneadings_weighted_sum_set = workerResult["kneadings_weighted_sum_set"] #достает числовой массив для визуализации
    kneadings_records = workerResult["kneadings_records"] #достает текстовые записи для файла

    param_x_caption = f"{grid_dict['first']['caption']}" #подпись оси x
    param_y_caption = f"{grid_dict['second']['caption']}" #подпись оси y

    param_x_count = left_n + right_n + 1 #количество точек по оси x
    param_y_count = up_n + down_n + 1 #количество точек по оси y

    #вычисляет реальные границы осей графика
    start_vals = {"gamma": gamma, "lambda": lam, "k": k}
    param_x_name = grid_dict["first"]["name"]
    param_y_name = grid_dict["second"]["name"]
    param_x_start = start_vals[param_x_name] - left_n * left_step
    param_x_end = start_vals[param_x_name] + right_n * right_step
    param_y_start = start_vals[param_y_name] - down_n * down_step
    param_y_end = start_vals[param_y_name] + up_n * up_step

    #построение карты режимов (каждому уникальному режиму присваивает свой цвет)
    plot_mode_map(
        kneadings_weighted_sum_set,
        set_random_color_map,
        param_x_caption,
        param_y_caption,
        param_x_start,
        param_x_end,
        param_x_count,
        param_y_start,
        param_y_end,
        param_y_count,
        font_size,
    )

    #добавление заголовка
    plt.title(fr"$\gamma={gamma}$, $\lambda={lam}$, $k={k}$", fontsize=font_size)

    #СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

    #NPY файл (бинарные данные)
    npy_outname = makeFinalOutname(config, initResult, "npy", startTime)
    np.save(npy_outname, kneadings_weighted_sum_set)
    print("Kneadings set successfully saved")

    #TXT файл (текстовые записи)
    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, "w", encoding="utf-8") as f:
        f.write(kneadings_records)
    print("Kneadings records successfully saved")

    #Изображение (PNG по умолчанию)
    img_extension = config["output"]["imageExtension"]
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    plt.savefig(plot_outname, dpi=600, bbox_inches="tight")
    plt.close()
    print("Mode map successfully saved")
