import numpy as np
from numba import cuda

#Отображает имена параметров в индексы массива. Это важно для CUDA, где мы работаем с числовыми массивами.
PARAM_TO_INDEX = {
    "gamma": 0,
    "lambda": 1,
    "k": 2,
}

DIM = 4 #Размерность фазового пространства
THREADS_PER_BLOCK = 512 #Оптимальное число потоков на блок для NVIDIA GPU

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
    dydt[1] = gamma - lam * v1 - np.sin(fi1) + k * np.sin(fi2 - fi1)

    dydt[2] = v2
    dydt[3] = gamma - lam * v2 - np.sin(fi2) + k * np.sin(fi1 - fi2)


@cuda.jit(device=True)
def stepper_rk4(params, y_curr, dt): #параметры системы, текущее состояние системы, шаг по времени
    #создание пяти массивов в локальной памяти CUDA-потока
    k1 = cuda.local.array(DIM, dtype=np.float64)
    k2 = cuda.local.array(DIM, dtype=np.float64)
    k3 = cuda.local.array(DIM, dtype=np.float64)
    k4 = cuda.local.array(DIM, dtype=np.float64)
    y_tmp = cuda.local.array(DIM, dtype=np.float64)

    rhs(params, y_curr, k1) #вычисляем k1
    """
    Если y_curr = [fi1=1.0, v1=0.5, fi2=1.2, v2=0.3], то:
    k1[0] = v1 = 0.5
    k1[1] = gamma - lam*v1 - sin(fi1) + k*sin(fi2-fi1)
    k1[2] = v2 = 0.3
    k1[3] = gamma - lam*v2 - sin(fi2) + k*sin(fi1-fi2)
    """

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


#пересекла ли некоторая величина ноль снизу вверх между двумя последовательными моментами времени
@cuda.jit(device=True)
def crossed_neg_to_pos(a_prev, a_curr):
    return (a_prev < -EPS) and (a_curr > EPS)

#пересекла ли некоторая величина ноль сверху вниз между двумя последовательными моментами времени
@cuda.jit(device=True)
def crossed_pos_to_neg(a_prev, a_curr):
    return (a_prev > EPS) and (a_curr < -EPS)


# Event symbols 0..7:
#  0 max fi1  : v1 + -> -
#  1 min fi1  : v1 - -> +
#  2 jump up  : -sin(fi1) - -> + и cos(fi1/2) меняет знак
#  3 jump down: -sin(fi1) + -> - и cos(fi1/2) меняет знак
#  4 max fi2  : v2 + -> -
#  5 min fi2  : v2 - -> +
#  6 jump up  : -sin(fi2) - -> + и cos(fi2/2) меняет знак
#  7 jump down: -sin(fi2) + -> - и cos(fi2/2) меняет знак

@cuda.jit(device=True)
def detect_event(y_prev, y_curr, out_evt):
    """
    out_evt[0] = 1 если событие произошло иначе 0
    out_evt[1] = символ 0..7 (только если событие произошло)
    """

    v1_prev = y_prev[1] #скорость первого маятника в предыдущий момент
    v1_curr = y_curr[1] #скорость первого маятника сейчас
    v2_prev = y_prev[3]
    v2_curr = y_curr[3]

    c1_prev = np.cos(0.5 * y_prev[0]) #cos(fi1/2) в предыдущий момент
    c1_curr = np.cos(0.5 * y_curr[0]) #cos(fi1/2) сейчас
    c2_prev = np.cos(0.5 * y_prev[2])
    c2_curr = np.cos(0.5 * y_curr[2])

    s1_prev = -np.sin(y_prev[0]) # -sin(fi1) в предыдущий момент
    s1_curr = -np.sin(y_curr[0]) # -sin(fi1) сейчас
    s2_prev = -np.sin(y_prev[2])
    s2_curr = -np.sin(y_curr[2])

    cross1 = crossed_neg_to_pos(c1_prev, c1_curr) or crossed_pos_to_neg(c1_prev, c1_curr) #cos(fi1/2) меняет знак
    cross2 = crossed_neg_to_pos(c2_prev, c2_curr) or crossed_pos_to_neg(c2_prev, c2_curr)

    #ивенты для fi1
    #скорость была положительной, стала отрицательной
    if (v1_prev > EPS) and (v1_curr < -EPS):
        out_evt[0] = 1
        out_evt[1] = 0
        return
    #скорость была отрицательной, стала положительной
    if (v1_prev < -EPS) and (v1_curr > EPS):
        out_evt[0] = 1
        out_evt[1] = 1
        return
    #-sin(fi1) в предыдущий момент был отрицательный, а в текущий момент стал положительный и cos(fi1/2) сменил знак
    if cross1 and (s1_prev < -EPS) and (s1_curr > EPS):
        out_evt[0] = 1
        out_evt[1] = 2
        return
    # -sin(fi1) в предыдущий момент был положительный, а в текущий момент стал отрицательный и cos(fi1/2) сменил знак
    if cross1 and (s1_prev > EPS) and (s1_curr < -EPS):
        out_evt[0] = 1
        out_evt[1] = 3
        return

    #ивенты для fi2
    if (v2_prev > EPS) and (v2_curr < -EPS):
        out_evt[0] = 1
        out_evt[1] = 4
        return
    if (v2_prev < -EPS) and (v2_curr > EPS):
        out_evt[0] = 1
        out_evt[1] = 5
        return
    if cross2 and (s2_prev < -EPS) and (s2_curr > EPS):
        out_evt[0] = 1
        out_evt[1] = 6
        return
    if cross2 and (s2_prev > EPS) and (s2_curr < -EPS):
        out_evt[0] = 1
        out_evt[1] = 7
        return

    #ни одно из 8 событий не произошло
    out_evt[0] = 0
    out_evt[1] = -1


#представляем последовательность символов в виде одного числа с основанием 8 (помещаем символы после десятичной точки)
"""
kneading_index: индекс текущего события (0, 1, 2, ...)
kneadings_end: конечный индекс 
kneadings_weighted_sum: уже накопленная сумма (изначально ноль)
"""
@cuda.jit(device=True)
def kneading_encoder_base8(symbol, kneading_index, kneadings_end, kneadings_weighted_sum):
    power = (-kneading_index + kneadings_end + 1)
    return kneadings_weighted_sum + (symbol / (8.0 ** power))


# здесь происходит численное интегрирование и сбор нидинг-последовательности
def make_integrator_rk4():
    @cuda.jit(device=True)
    def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):

        y_prev = cuda.local.array(DIM, dtype=np.float64) #создаем новый массив
        for k in range(DIM):
            y_prev[k] = y_curr[k] #и его копию

        kneading_index = 0 #сколько событий уже произошло
        kneadings_weighted_sum = 0.0 #накопленная сумма

        evt = cuda.local.array(2, dtype=np.int32) #массив событий (будет хранить флаг события и код события)

        for _ in range(1, n): #начальное состояние уже есть, нужно сделать n-1 шагов, чтобы получить n состояний
            for __ in range(stride): #один шаг RK4 (параметр stride позволяет делать несколько шагов между проверками)
                stepper_rk4(params, y_curr, dt)

            #проверяет, не стали ли значения слишком большими. если да - возвращает код ошибки -0.2.
            for k in range(DIM):
                if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                    return InfinityError

            detect_event(y_prev, y_curr, evt) #анализирует, произошло ли какое-либо событие за этот шаг
            if evt[0] == 1: #если событие произошло
                if kneading_index >= kneadings_start: #если мы уже в интервале записи
                    kneadings_weighted_sum = kneading_encoder_base8(
                        evt[1], kneading_index, kneadings_end, kneadings_weighted_sum
                    ) #добавляем событие в сумму

                kneading_index += 1 #увеличиваем счетчик событий
                if kneading_index > kneadings_end: #если  набрали нужное количество событий
                    return kneadings_weighted_sum #возвращаем результат

            #подготовка к следующей интерации, текущее состояние становится предыдущим
            for k in range(DIM):
                y_prev[k] = y_curr[k]

        return KneadingDoNotEndError #цикл закончился, а нужное количество событий не набрано - возвращаем код ошибки
    return integrator_rk4


def make_sweep_threads():
    integrator_rk4 = make_integrator_rk4()

    @cuda.jit
    def sweep_threads(
        kneadings_weighted_sum_set,
        inits,
        nones,
        params_x,
        params_y,
        def_params,
        param_x_idx,
        param_y_idx,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end,
    ):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x #вычисляет уникальный ID каждого потока (от 0 до общего числа потоков)
        total = (left_n + right_n + 1) * (up_n + down_n + 1) #общее количество точек

        if idx >= total: #потоков может быть немного больше, чем точек сетки. Лишние потоки ничего не делают.
            return

        #проверяет, не помечена ли эта точка как пропущенная. если да - записывает код ошибки -0.3 и выходит
        for i in range(len(nones)):
            if idx == nones[i]:
                kneadings_weighted_sum_set[idx] = NoInitFound
                return

        #загрузка начальных условий
        y = cuda.local.array(DIM, dtype=np.float64) # локальный массив для состояния
        base = idx * DIM # где в плоском массиве начинаются данные для этой точки (точка 0 - индексы 0-3, точка 1 - индексы 4-7 и тд)
        for k in range(DIM):
            y[k] = inits[base + k] # копируем [fi1, v1, fi2, v2]

        #загрузка фиксированных параметров системы в локальную память потока
        params = cuda.local.array(3, dtype=np.float64) # Создаем локальный массив для 3 параметров
        for k in range(3):
            params[k] = def_params[k] # Копируем значения из def_params (глобальный массив) в params (локальный)

        #заменяем фиксированные значения параметров на индивидуальные для данной точки сетки
        params[param_x_idx] = params_x[idx]
        params[param_y_idx] = params_y[idx]

        #запуск интергратора
        kneadings_weighted_sum_set[idx] = integrator_rk4(
            y, params, dt, n, stride, kneadings_start, kneadings_end
        )

    return sweep_threads #возвращает ядро


def sweep(
    inits,
    nones,
    params_x,
    params_y,
    def_params,
    param_to_index,
    param_x_str,
    param_y_str,
    up_n,
    down_n,
    left_n,
    right_n,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    total = (left_n + right_n + 1) * (up_n + down_n + 1) #количество точек

    #Host-буфер
    kneadings_weighted_sum_set = np.zeros(total, dtype=np.float64) #создает пустой массив на CPU для результатов

    #Device-буфер
    kneadings_weighted_sum_set_gpu = cuda.device_array(total, dtype=np.float64) #выделяет память на GPU для результатов

    #Копирование данных на GPU (Каждая строка: преобразует входные данные в numpy-массив нужного типа, копирует их из оперативной памяти в видеопамять)
    inits_gpu = cuda.to_device(np.asarray(inits, dtype=np.float64))
    nones_gpu = cuda.to_device(np.asarray(nones, dtype=np.int32))
    def_params_gpu = cuda.to_device(np.asarray(def_params, dtype=np.float64))
    params_x_gpu = cuda.to_device(np.asarray(params_x, dtype=np.float64))
    params_y_gpu = cuda.to_device(np.asarray(params_y, dtype=np.float64))

    #Использует словарь для получения числовых индексов параметров
    param_x_idx = param_to_index[param_x_str]
    param_y_idx = param_to_index[param_y_str]

    # Вычисление количества блоков (округление вверх при делении)
    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"Num of blocks per grid:       {blocks}")
    print(f"Num of threads per block:     {THREADS_PER_BLOCK}")
    print(f"Total Num of threads running: {blocks * THREADS_PER_BLOCK}")

    """
    Создается ядро sweep_threads
    Запускается на 706 (например) блоках по 512 потоков
    Каждый поток получает свои аргументы
    Все 361472 потока начинают работу одновременно
    """

    sweep_threads = make_sweep_threads()
    sweep_threads[blocks, THREADS_PER_BLOCK](
        kneadings_weighted_sum_set_gpu, #куда писать результат
        inits_gpu, #начальные условия
        nones_gpu, #пропущенные точки
        params_x_gpu, #значения параметра x
        params_y_gpu, #значения параметра y
        def_params_gpu, #фиксированные параметры
        param_x_idx, #индекс параметра x
        param_y_idx, #индекс параметра y
        up_n, # размеры сетки
        down_n,
        left_n,
        right_n,
        dt, #параметры интегрирования
        n,
        stride,
        kneadings_start, #границы записи
        kneadings_end,
    )

    kneadings_weighted_sum_set_gpu.copy_to_host(kneadings_weighted_sum_set) #Пересылает результаты из видеопамяти обратно в оперативную память
    return kneadings_weighted_sum_set #Возвращает массив с результатами для всех точек сетки
