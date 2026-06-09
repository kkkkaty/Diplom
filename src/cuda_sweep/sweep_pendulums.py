import numpy as np
from numba import cuda
import math

#Отображает имена параметров в индексы массива. Это важно для CUDA, где мы работаем с числовыми массивами.
PARAM_TO_INDEX = {
    "gamma": 0,
    "lambda": 1,
    "k": 2,
}

DIM = 4 #Размерность фазового пространства
THREADS_PER_BLOCK = 256 #Оптимальное число потоков на блок для NVIDIA GPU

INFINITY = 1e6 #Порог "взрыва" решения
EPS = 1e-12  #Маленькое число для сравнения с нулем

#Коды ошибок (отрицательные, чтобы не путать с нормальными результатами)
KneadingDoNotEndError = -0.1 #Серия не закончилась за время интегрирования
InfinityError = -0.2 #Решение "взорвалось"
NoInitFound = -0.3 #Точка пропущена (nones)

#система
@cuda.jit(device=True)
def rhs(params, y, dydt):
    gamma = params[0]
    lam = params[1]
    k = params[2]

    fi1 = y[0]
    v1 = y[1]
    fi2 = y[2]
    v2 = y[3]

    dydt[0] = v1
    dydt[1] = gamma - lam * v1 - math.sin(fi1) + k * math.sin(fi2 - fi1)

    dydt[2] = v2
    dydt[3] = gamma - lam * v2 - math.sin(fi2) + k * math.sin(fi1 - fi2)


@cuda.jit(device=True)
def stepper_rk4(params, y_curr, dt): #параметры системы, текущее состояние системы, шаг по времени
    #создание пяти массивов в локальной памяти CUDA-потока
    k1 = cuda.local.array(DIM, dtype=np.float64)
    k2 = cuda.local.array(DIM, dtype=np.float64)
    k3 = cuda.local.array(DIM, dtype=np.float64)
    k4 = cuda.local.array(DIM, dtype=np.float64)
    y_tmp = cuda.local.array(DIM, dtype=np.float64)

    rhs(params, y_curr, k1) #вычисляем k1

    #вычисляем k2
    for i in range(DIM):
        y_tmp[i] = y_curr[i] + 0.5 * dt * k1[i]
    rhs(params, y_tmp, k2)

    # вычисляем k3
    for i in range(DIM):
        y_tmp[i] = y_curr[i] + 0.5 * dt * k2[i]
    rhs(params, y_tmp, k3)

    # вычисляем k4
    for i in range(DIM):
        y_tmp[i] = y_curr[i] + dt * k3[i]
    rhs(params, y_tmp, k4)

    #финальное обновление y_curr
    for i in range(DIM):
        y_curr[i] = y_curr[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    # Заворачиваем фазу для точности
    for i in (0, 2):
        if y_curr[i] > math.pi: y_curr[i] -= 2.0 * math.pi
        if y_curr[i] < -math.pi: y_curr[i] += 2.0 * math.pi


#пересекла ли некоторая величина ноль снизу вверх между двумя последовательными моментами времени
@cuda.jit(device=True)
def crossed_neg_to_pos(a_prev, a_curr):
    return (a_prev < -EPS) and (a_curr > EPS)

#пересекла ли некоторая величина ноль сверху вниз между двумя последовательными моментами времени
@cuda.jit(device=True)
def crossed_pos_to_neg(a_prev, a_curr):
    return (a_prev > EPS) and (a_curr < -EPS)

# Определяет событие для ОДНОГО маятника
#  0 max fi  : v + -> -
#  1 min fi  : v - -> +
#  2 jump up  : -sin(fi) - -> + и -(1+sin(fi)+cos(fi)) - -> +
#  3 jump down: -sin(fi) + -> - и -(1+sin(fi)+cos(fi)) + -> -
#  4 no Event

@cuda.jit(device=True)
def detect_event(y_prev, y_curr, out_evt):
    # Подготовка переменных для маятника 1
    v1_prev = y_prev[1]
    v1_curr = y_curr[1]
    phi1_prev = y_prev[0]
    phi1_curr = y_curr[0]

    s1_prev = math.sin(phi1_prev)
    s1_curr = math.sin(phi1_curr)
    f1_prev = 1.0 + math.cos(phi1_prev) + s1_prev
    f1_curr = 1.0 + math.cos(phi1_curr) + s1_curr

    # Подготовка переменных для маятника 2
    v2_prev = y_prev[3]
    v2_curr = y_curr[3]
    phi2_prev = y_prev[2]
    phi2_curr = y_curr[2]

    s2_prev = math.sin(phi2_prev)
    s2_curr = math.sin(phi2_curr)
    f2_prev = 1.0 + math.cos(phi2_prev) + s2_prev
    f2_curr = 1.0 + math.cos(phi2_curr) + s2_curr

    # Логика для маятника 1
    out_evt[0] = 4  # по умолчанию нет ивента
    if (v1_prev > EPS) and (v1_curr < -EPS):
        out_evt[0] = 0
    elif (v1_prev < -EPS) and (v1_curr > EPS):
        out_evt[0] = 1
    elif (-s1_prev < -EPS and -s1_curr > EPS) and (-f1_prev < -EPS and -f1_curr > EPS):
        out_evt[0] = 2
    elif (-s1_prev > EPS and -s1_curr < -EPS) and (-f1_prev > EPS and -f1_curr < -EPS):
        out_evt[0] = 3

    # Логика для маятника 2
    out_evt[1] = 4  # по умолчанию нет ивента
    if (v2_prev > EPS) and (v2_curr < -EPS):
        out_evt[1] = 0
    elif (v2_prev < -EPS) and (v2_curr > EPS):
        out_evt[1] = 1
    elif (-s2_prev < -EPS and -s2_curr > EPS) and (-f2_prev < -EPS and -f2_curr > EPS):
        out_evt[1] = 2
    elif (-s2_prev > EPS and -s2_curr < -EPS) and (-f2_prev > EPS and -f2_curr < -EPS):
        out_evt[1] = 3

    out_evt[2] = out_evt[0] * 5 + out_evt[1]
#00-max fi1, max fi2
#01-max fi1, min fi2
#02-max fi1, jump up fi2
#03-max fi1, jump down fi2
#04-max fi1, no events
#05-min fi1, max fi2
#06-min fi1, min fi2
#07-min fi1, jump up fi2
#08-min fi1, jump down fi2
#09-min fi1, no events
#10-jump up fi1, max fi2
#11-jump up fi1, min fi2
#12-jump up fi1, jump up fi2
#13-jump up fi1, jump down fi2
#14-jump up fi1, no events
#15-jump down fi1, max fi2
#16-jump down fi1, min fi2
#17-jump down fi1, jump up fi2
#18-jump down fi1, jump down fi2
#19-jump down fi1, no events
#20-no events, max fi2
#21-no events, min fi2
#22-no events, jump up fi2
#23-no events, jump down fi2
#24-no events, no events


#представляем последовательность символов в виде одного числа с основанием 25 (помещаем символы после десятичной точки)
"""
kneading_index: индекс текущего события (0, 1, 2, ...)
kneadings_end: конечный индекс 
kneadings_weighted_sum: уже накопленная сумма (изначально ноль)
"""
@cuda.jit(device=True)
def kneading_encoder_base25(symbol, kneading_index,  kneadings_weighted_sum):
    power = kneading_index + 1
    return kneadings_weighted_sum + (symbol / (25.0 ** power))


#Ищет минимальный период в последовательности символов
@cuda.jit(device=True)
def find_period_of_sequence(sequence, length):
    check_len = length // 2 # Берем только хвост (половину последовательности), чтобы забыть переходный процесс
    if check_len < 2:
        return 0
    max_p = check_len // 2  # Ограничиваем максимальный период
    for p in range(1, max_p + 1): #перебор всех периодов
        match = True #предполагаем, что период найден, пока не нашли противоречие
        for i in range(length - 1, length - check_len + p, -1): #цикл по хвосту справа налево
            if sequence[i] != sequence[i - p]: #совпадает ли символ с символом p шагов назад
                match = False #если нет, то период разрушен
                break
        if match: return p #возвращаем период
    return 0

@cuda.jit(device=True)
def analyze_attractor_period(sequence, length, p):
    has_osc1 = False
    has_rot1 = False
    has_osc2 = False
    has_rot2 = False

    n1_pos = 0
    n1_neg = 0
    n2_pos = 0
    n2_neg = 0

    start_idx = length - p
    if start_idx < 0:
        start_idx = 0
    for i in range(start_idx, length):
        symbol = sequence[i]
        m1 = symbol // 5
        m2 = symbol % 5

        # Маятник 1
        if m1 == 0 or m1 == 1:
            has_osc1 = True
        elif m1 == 2:
            has_rot1 = True
            n1_pos += 1
        elif m1 == 3:
            has_rot1 = True
            n1_neg += 1

        # Маятник 2
        if m2 == 0 or m2 == 1:
            has_osc2 = True
        elif m2 == 2:
            has_rot2 = True
            n2_pos += 1
        elif m2 == 3:
            has_rot2 = True
            n2_neg += 1


    # Ограничиваем счетчики, чтобы они не превышали 999
    # и не "перетекали" в целую часть (индекс режима)
    if n1_pos > 999: n1_pos = 999
    if n1_neg > 999: n1_neg = 999
    if n2_pos > 999: n2_pos = 999
    if n2_neg > 999: n2_neg = 999

    # Определяем типы
    t1 = 0
    if has_rot1:
        t1 = 2
    elif has_osc1:
        t1 = 1

    t2 = 0
    if has_rot2:
        t2 = 2
    elif has_osc2:
        t2 = 1

    mode = float(t1 * 3 + t2)

    # Собираем итоговое число
    res = mode
    res = res + float(n1_pos) * 0.001
    res = res + float(n1_neg) * 0.000001
    res = res + float(n2_pos) * 0.000000001
    res = res + float(n2_neg) * 0.000000000001
    return res

@cuda.jit(device=True)
def integrator_regime_analysis(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):
    y_prev = cuda.local.array(DIM, dtype=np.float64) #хранит прошлое состояние системы
    for k in range(DIM): y_prev[k] = y_curr[k] #копируем текущее состояние
    seq = cuda.local.array(1024, dtype=np.int32) #храним последовательность

    evt_index = 0
    for step in range(n):
        for _ in range(stride):
            stepper_rk4(params, y_curr, dt)

        if abs(y_curr[0]) > INFINITY: return -0.2

        evt = cuda.local.array(3, dtype=np.int32) #массив событий
        detect_event(y_prev, y_curr, evt) #поиск событий

        if evt[0] != 4 or evt[1] != 4: #если хотя бы для одного из маятников есть настоящее событие
            if evt_index >= kneadings_start and evt_index < kneadings_end:
                write_idx = evt_index - kneadings_start
                if write_idx < 1024: seq[write_idx] = evt[2]
            evt_index += 1
            if evt_index >= kneadings_end: break #если собрали достаточно, заканчиваем

        for k in range(DIM): y_prev[k] = y_curr[k] #текущее состояние становится предыдущим

    # Если событий мало — считаем регулярным режимом
    if evt_index < kneadings_end:
        return 1.0

    length = min(1024, kneadings_end - kneadings_start) #длина последовательности
    period = find_period_of_sequence(seq, length) #пытаемся найти период

    #1.0 — порядок, 0.0 — хаос
    if period > 0:
        return 1.0
    else:
        return 0.0


# здесь происходит численное интегрирование и сбор нидинг-последовательности
def make_integrator_rk4():
    @cuda.jit(device=True)
    def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):

        y_prev = cuda.local.array(DIM, dtype=np.float64) #создаем новый массив
        for k in range(DIM):
            y_prev[k] = y_curr[k] #и его копию

        kneading_index = 0 #сколько событий уже произошло
        kneadings_weighted_sum = 0.0 #накопленная сумма

        evt = cuda.local.array(3, dtype=np.int32) #массив событий (будет хранить флаг события и код события)

        for _ in range(1, n): #начальное состояние уже есть, нужно сделать n-1 шагов, чтобы получить n состояний
            for __ in range(stride): #один шаг RK4 (параметр stride позволяет делать несколько шагов между проверками)
                stepper_rk4(params, y_curr, dt)

            #проверяет, не стали ли значения слишком большими. если да - возвращает код ошибки -0.2.
            for k in range(DIM):
                if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                    return InfinityError

            detect_event(y_prev, y_curr, evt) #анализирует, произошло ли какое-либо событие за этот шаг
            if evt[0] != 4 or evt[1] != 4: #если событие произошло
                if kneading_index >= kneadings_start: #если мы уже в интервале записи
                    kneadings_weighted_sum = kneading_encoder_base25(evt[2], kneading_index- kneadings_start, kneadings_weighted_sum) #добавляем событие в сумму

                kneading_index += 1 #увеличиваем счетчик событий
                if kneading_index > kneadings_end: #если  набрали нужное количество событий
                    return kneadings_weighted_sum #возвращаем результат

            #подготовка к следующей интерации, текущее состояние становится предыдущим
            for k in range(DIM):
                y_prev[k] = y_curr[k]

        return KneadingDoNotEndError #цикл закончился, а нужное количество событий не набрано - возвращаем код ошибки
    return integrator_rk4


def make_sweep_threads(use_regime_mode=False, use_9color_mode=False):
    integrator_standard = make_integrator_rk4()

    @cuda.jit
    def sweep_threads(
            kneadings_weighted_sum_set, inits, nones, params_x, params_y,
            def_params, param_x_idx, param_y_idx, up_n, down_n, left_n, right_n,
            dt, n, stride, kneadings_start, kneadings_end,
    ):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        total = (left_n + right_n + 1) * (up_n + down_n + 1)
        if idx >= total: return

        for i in range(len(nones)):
            if idx == nones[i]:
                kneadings_weighted_sum_set[idx] = NoInitFound
                return

        y = cuda.local.array(DIM, dtype=np.float64)
        base = idx * DIM
        for k in range(DIM): y[k] = inits[base + k]

        params = cuda.local.array(3, dtype=np.float64)
        for k in range(3): params[k] = def_params[k]
        params[param_x_idx] = params_x[idx]
        params[param_y_idx] = params_y[idx]

        # ЛОГИКА ВЫБОРА РЕЖИМА
        if use_regime_mode:
            res = integrator_regime_analysis(y, params, dt, n, stride, kneadings_start, kneadings_end)
        else:
            res = integrator_standard(y, params, dt, n, stride, kneadings_start, kneadings_end)

        kneadings_weighted_sum_set[idx] = res

    return sweep_threads

@cuda.jit
def sweep_continuation_kernel(
        results_col, current_states, next_states,
        p_x_col, p_y_col, def_params, p_x_idx, p_y_idx,
        n_settle, n_analyze, dt, stride,
        kneadings_threshold
):
    idx = cuda.grid(1)
    if idx >= results_col.shape[0]: return

    y = cuda.local.array(DIM, dtype=np.float64)
    y_prev = cuda.local.array(DIM, dtype=np.float64)
    seq = cuda.local.array(1024, dtype=np.int32)
    params = cuda.local.array(3, dtype=np.float64)
    evt = cuda.local.array(3, dtype=np.int32)

    for k in range(3): params[k] = def_params[k]
    params[p_x_idx], params[p_y_idx] = p_x_col[idx], p_y_col[idx]
    for k in range(DIM): y[k] = current_states[idx, k]

    for _ in range(n_settle):
        for __ in range(stride): stepper_rk4(params, y, dt)

    # Сбор символов
    evt_index = 0
    for step in range(n_analyze):
        for k in range(DIM): y_prev[k] = y[k]
        for _ in range(stride): stepper_rk4(params, y, dt)

        detect_event(y_prev, y, evt)
        if evt[0] != 4 or evt[1] != 4:
            if evt_index < 1024:
                seq[evt_index] = evt[2]
                evt_index += 1
            else:
                break
        if abs(y[0]) > INFINITY: break


    if evt_index == 0:
        results_col[idx] = 0.0  # Покой
    else:
        recorded_len = min(evt_index, 1024)
        # Сначала пытаемся найти период
        p = find_period_of_sequence(seq, recorded_len)

        if p > 0:
            # Если период найден
            # Считаем N1, N2 за ОДИН цикл
            results_col[idx] = analyze_attractor_period(seq, recorded_len, p)

        else:
            # Если период не найден
            if evt_index < kneadings_threshold:
                # Случай 1: Событий мало. Это не хаос, просто медленная точка.
                results_col[idx] = analyze_attractor_period(seq, recorded_len, recorded_len)
                # В логах здесь N1 будет большим, но цвет на карте будет верным.

            else:
                # Случай 2: Событий много, а периода нет -> хаос
                results_col[idx] = -1.0

    # Сохраняем y для протягивания
    for k in range(DIM): next_states[idx, k] = y[k]


def sweep(
        inits, nones, params_x, params_y, def_params, param_to_index,
        param_x_str, param_y_str, up_n, down_n, left_n, right_n,
        dt, n, stride, kneadings_start, kneadings_end,
        use_regime_mode=False, use_9color_mode=False, use_continuation_mode=False,
):
    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    total = cols * rows
    results = np.zeros(total, dtype=np.float64)

    # 1. Готовим итоговый массив на ГПУ
    results_gpu = cuda.device_array(total, dtype=np.float64)
    def_params_gpu = cuda.to_device(np.asarray(def_params, dtype=np.float64))
    param_x_idx = param_to_index[param_x_str]
    param_y_idx = param_to_index[param_y_str]

    if use_continuation_mode:
        print("ЗАПУСК: Режим протягивания аттрактора (постолбцово)")

        # Подготовка состояний: берем самый левый столбец для старта (c_idx = 0). Берем сепаратрисные начальные условия только для 1-го столбца
        # (current_states будет хранить текущую точку аттрактора для каждой строки)
        current_states = inits.reshape(rows, cols, DIM)[:, 0, :].copy()
        d_states_in = cuda.to_device(current_states)
        d_states_out = cuda.device_array_like(d_states_in)

        # Разрезаем сетку параметров на 2D для удобства доступа к столбцам
        px_2d = params_x.reshape(rows, cols)
        py_2d = params_y.reshape(rows, cols)

        blocks = (rows + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        # Настройки времени
        n_settle_long = int(kneadings_start)  # Для первого столбца
        n_settle_short = int(kneadings_start // 10)  # Для последующих
        n_analyze = 300000  # Время сбора символов
        threshold = int(kneadings_end)  # Лимит детекции хаоса

        for c_idx in range(cols):
            # Отправляем на ГПУ параметры только одного столбца
            d_px = cuda.to_device(np.ascontiguousarray(px_2d[:, c_idx]))
            d_py = cuda.to_device(np.ascontiguousarray(py_2d[:, c_idx]))

            # Создаем временный буфер для результатов одного столбца
            d_res_col = cuda.device_array(rows, dtype=np.float64)

            # Выбираем settle: для первого столбца - долгий, для остальных - короткий
            settle = n_settle_long if c_idx == 0 else n_settle_short

            # запуск ядра одного столбца
            sweep_continuation_kernel[blocks, THREADS_PER_BLOCK](
                d_res_col, d_states_in, d_states_out,
                d_px, d_py, def_params_gpu, param_x_idx, param_y_idx,
                settle, n_analyze, dt, stride,
                threshold
            )

            res_host = d_res_col.copy_to_host()
            results[c_idx::cols] = res_host

            # Протягиваем аттрактор: выход этого столбца становится входом следующего
            d_states_in.copy_to_device(d_states_out)
            if c_idx % 20 == 0:
                print(f"Столбец {c_idx}/{cols} готов...")

        return results  # Возвращаем заполненный массив

    else:
        #для нидингов и ЧБ
        inits_gpu = cuda.to_device(np.asarray(inits, dtype=np.float64))
        nones_gpu = cuda.to_device(np.asarray(nones, dtype=np.int32))
        params_x_gpu = cuda.to_device(np.asarray(params_x, dtype=np.float64))
        params_y_gpu = cuda.to_device(np.asarray(params_y, dtype=np.float64))

        blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        sweep_threads = make_sweep_threads(use_regime_mode=use_regime_mode, use_9color_mode=use_9color_mode)
        sweep_threads[blocks, THREADS_PER_BLOCK](
            results_gpu, inits_gpu, nones_gpu, params_x_gpu, params_y_gpu,
            def_params_gpu, param_x_idx, param_y_idx, up_n, down_n, left_n, right_n,
            dt, n, stride, kneadings_start, kneadings_end
        )
        return results_gpu.copy_to_host()