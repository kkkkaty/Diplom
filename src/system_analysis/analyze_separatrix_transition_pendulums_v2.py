#анализатор перехода между двумя областями на карте нидингов

#Берёт уже посчитанную карту нидингов
#Выбирает горизонтальную строку карты
#Ищет место, где код нидинга меняется: например слева код 0123, справа 0456
#Считает, что между этими двумя точками находится граница перехода
#Берёт три точки около границы:слева от границы, прямо на границе, справа от границы
#В каждой из трёх точек параметров строит начальное условие около сепаратрисы
#Интегрирует траекторию
#Проверяет, к какому равновесию траектория ближе всего подходит
#Сохраняет отчёт .json, текстовый отчёт и картинки

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.system_analysis.get_inits import (
    build_separatrix_init_for_point, #строит начальное условие около сепаратрисы
    equilibrium_type, #определяет тип равновесия
    find_equilibria_pendulum, #ищет состояния равновесия системы
    rk4_step, #один шаг метода Рунге–Кутты 4-го порядка
)

EPS_LOG = 1e-30 #Очень маленькое число, чтобы не брать логарифм от нуля


#Это одна точка на горизонтальном срезе карты
@dataclass
class ScanPoint:
    i: int #номер столбца
    j: int #номер строки
    x: float #значение параметра по горизонтальной оси
    y: float #значение параметра по вертикальной оси
    raw_value: float #число, которое хранится в карте нидингов
    code: str #расшифрованный код нидинга в виде строки


#Это кандидат на равновесие
@dataclass
class EquilibriumCandidate:
    point: np.ndarray #координаты равновесия
    nU: int #количество неустойчивых направлений
    nS: int #количество устойчивых направлений
    nC: int #количество центральных направлений
    eigvals: np.ndarray #собственные значения матрицы
    is_symmetric: bool #является ли равновесие симметричным


#Это результат для одной пробной точки
@dataclass
class ProbeResult:
    label: str  # название (слева или справа от границы)
    side: str   # какая строна
    x_param: float #координаты точки на карте параметров
    y_param: float
    params: Tuple[float, float, float] #полный набор параметров системы
    source_eq: np.ndarray #Равновесие, из которого выходит сепаратриса
    init_point: np.ndarray #Точка, с которой реально запускается интегрирование (находится рядом с source_eq вдоль неуст напр)
    unstable_dir: np.ndarray #Неустойчивый собственный вектор
    branch_id: int #Какая ветвь сепаратрисы выбрана
    trajectory: np.ndarray #Вся рассчитанная траектория
    time: np.ndarray #Массив времени
    closest_eq: np.ndarray #Равновесие, к которому траектория подошла ближе всего
    closest_eq_meta: Dict[str, Any]  #Информация о closest_eq: тип, собственные значения, симметричность
    log_distance_to_best_eq: np.ndarray #график log10(||x(t)-x(eq)||+10^(-30))
    best_eq_min_log10: float #Минимальное значение этого графика. Чем меньше, тем ближе траектория подошла к равновесию.
    best_eq_argmin_t: float #Время, в которое расстояние было минимальным.
    best_eq_argmin_index: int #Индекс этого момента в массиве
    candidate_eqs: List[EquilibriumCandidate]


#Это итог всего анализа одной границы
@dataclass
class TransitionAnalysisResult:
    row_index: int #Номер горизонтальной строки карты
    transition_left_index: int #Индексы двух соседних точек карты, между которыми поменялся код
    transition_right_index: int
    left_scan_point: ScanPoint #Сами точки слева и справа
    right_scan_point: ScanPoint
    x_boundary_estimate: float #Оценка положения границы =(x(left)+x(right))/2
    y_scan_value: float #Значение второго параметра на выбранной горизонтальной линии
    probes: List[ProbeResult] #результаты before, after
    candidate_equilibria: List[Dict[str, Any]] #Все найденные равновесия-кандидаты
    output_dir: str #Папка, куда сохраняются графики и отчёты
    common_eq: Optional[np.ndarray] = None
    common_eq_meta: Optional[Dict[str, Any]] = None



#Вспомогательные функции

#Эта функция сворачивает угол по модулю 2pi
def wrap_angle_0_2pi(phi: np.ndarray | float) -> np.ndarray | float:
    return np.mod(phi, 2.0 * np.pi)

#Расшифровывает raw kneading value в строку символов
def decode_base8_weighted(x: float, length: int) -> str:
    if x < 0:
        return str(x)
    s: List[str] = []
    v = float(x)
    for _ in range(length):
        v *= 8.0
        d = int(v + 1e-12)
        d = max(0, min(7, d))
        s.append(str(d))
        v -= d
    return "".join(s)


#Эта функция строит сетку значений двух параметров
def _generate_parameters_2d(
    start_x: float,
    start_y: float,
    up_n: int,
    down_n: int,
    left_n: int,
    right_n: int,
    up_step: float,
    down_step: float,
    left_step: float,
    right_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
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


#Эта функция берёт всё нужное из YAML-конфига
def build_parameter_grid_from_config(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    def_sys = config["defaultSystem"]
    grid = config["grid"]

    gamma = float(def_sys["gamma"])
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])
    start_vals = {"gamma": gamma, "lambda": lam, "k": k}

    param_x_name = grid["first"]["name"]
    param_y_name = grid["second"]["name"]

    left_n = int(grid["first"]["left_n"])
    right_n = int(grid["first"]["right_n"])
    down_n = int(grid["second"]["down_n"])
    up_n = int(grid["second"]["up_n"])

    params_x, params_y = _generate_parameters_2d(
        start_x=start_vals[param_x_name],
        start_y=start_vals[param_y_name],
        up_n=up_n,
        down_n=down_n,
        left_n=left_n,
        right_n=right_n,
        up_step=float(grid["second"]["up_step"]),
        down_step=float(grid["second"]["down_step"]),
        left_step=float(grid["first"]["left_step"]),
        right_step=float(grid["first"]["right_step"]),
    )

    cols = left_n + right_n + 1
    rows = up_n + down_n + 1
    return params_x, params_y, cols, rows



def reshape_map(values: np.ndarray, cols: int, rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64) ##Преобразуем карту в numpy-массив
    if arr.size != cols * rows:
        raise ValueError(f"Map size mismatch: got {arr.size}, expected {cols * rows}")
    return arr.reshape(rows, cols) #Плоский массив превращается в матрицу


#Эта функция берёт одну горизонтальную линию карты
def extract_horizontal_scan(
    kneading_map_flat: np.ndarray,
    config: Dict[str, Any],
    row_index: Optional[int] = None,
) -> List[ScanPoint]:
    params_x, params_y, cols, rows = build_parameter_grid_from_config(config) #Строит сетку параметров
    arr2d = reshape_map(kneading_map_flat, cols, rows) #Преобразует карту в двумерный вид

    if row_index is None: #Если строка не задана, берётся центральная строка
        row_index = int(config["grid"]["second"]["down_n"])
    if row_index < 0 or row_index >= rows: #Проверка, что строка существует
        raise ValueError(f"row_index={row_index} out of range [0, {rows - 1}]")

    knead_cfg = config["kneadings_pendulums"] #Вычисляется длина кода нидинга
    seq_len = int(knead_cfg["kneadings_end"]) - int(knead_cfg["kneadings_start"]) + 1

    out: List[ScanPoint] = [] #Создаётся список точек среза
    for i in range(cols): #Идём по всем столбцам выбранной строки
        idx = i + row_index * cols #Получаем индекс в плоском массиве
        out.append( #Создаётся объект с:индексом; реальным параметром; raw value; расшифрованным кодом.
            ScanPoint(
                i=i,
                j=row_index,
                x=float(params_x[idx]),
                y=float(params_y[idx]),
                raw_value=float(arr2d[row_index, i]),
                code=decode_base8_weighted(float(arr2d[row_index, i]), seq_len),
            )
        )
    return out #Возвращается горизонтальный срез карты


#Функция ищет места, где код нидинга меняется
def find_code_transitions_on_scan(scan: Sequence[ScanPoint]) -> List[Tuple[int, int]]:
    transitions: List[Tuple[int, int]] = [] #Список переходов
    for left_i in range(len(scan) - 1): #Идём по соседним точкам
        a = scan[left_i] #Берём левую и правую точки
        b = scan[left_i + 1]
        if a.raw_value < 0 or b.raw_value < 0: #Если одно из значений отрицательное, переход не рассматриваем
            continue
        if a.code != b.code: #Если коды разные, значит между этими точками есть граница областей
            transitions.append((left_i, left_i + 1))
    return transitions # Возвращаем список найденных переходов



# Интегрирование траектории
def integrate_trajectory(
    y0: np.ndarray, #Копия начального состояния
    gamma: float,
    lam: float,
    k: float,
    dt: float,
    n_steps: int,
    stride: int = 1,
    infinity_threshold: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array(y0, dtype=float).copy()
    traj = np.empty((n_steps + 1, 4), dtype=float) #Массивы для траектории и времени
    t = np.empty(n_steps + 1, dtype=float)
    traj[0] = y #Записываем начальную точку
    t[0] = 0.0

    eff_dt = dt * stride #Если stride > 1, то сохраняются не все маленькие шаги, а каждый stride-й
    for step in range(1, n_steps + 1): #Основной цикл интегрирования
        for _ in range(stride): #Делаем stride шагов методом Рунге–Кутты
            y = rk4_step(y, dt, gamma, lam, k)
        traj[step] = y #Сохраняем точку траектории и время
        t[step] = step * eff_dt
        if np.any(np.abs(y) > infinity_threshold): #Если траектория улетела слишком далеко, интегрирование досрочно останавливается
            return t[: step + 1], traj[: step + 1]
    return t, traj #Если всё нормально, возвращаем полную траекторию


#возвращает fi2, v2, fi1, v1
def swap_pendulums(eq: np.ndarray) -> np.ndarray:
    return np.array([eq[2], eq[3], eq[0], eq[1]], dtype=float)


#найти все равновесия
#определить их тип
#понять, симметричные они или нет
def find_candidate_equilibria(gamma: float, lam: float, k: float) -> List[EquilibriumCandidate]:
    eqs = find_equilibria_pendulum(gamma, k) #все стационарные решения системы
    eqs = [np.asarray(eq, dtype=float) for eq in eqs] #приводим всё к numpy-массивам

    candidates: List[EquilibriumCandidate] = []

    for eq in eqs: #Перебираем каждое равновесие
        info = equilibrium_type(eq, gamma, lam, k) #число неустойчивых направлений, устойчивых, центральных, собственные значения
        is_diagonal_symmetric = bool( #проверка, что это синхронное симметричное равновесие (fi1=fi2, v1=v2=0)
            np.isclose(eq[0], eq[2], atol=1e-9)
            and np.isclose(eq[1], 0.0, atol=1e-9)
            and np.isclose(eq[3], 0.0, atol=1e-9)
        )
        swapped = swap_pendulums(eq) #перестановка маятников

        #существует ли равновесие fi2, v2, fi1, v1 (симметричное начальному fi1, v1, fi2, v2)
        has_swapped_partner = any(
            np.linalg.norm(swapped - other) < 1e-8
            for other in eqs)

        candidates.append( #сохраняем координату, тип равновесия, собственные значения и симметричность
            EquilibriumCandidate(
                point=eq,
                nU=int(info["nU"]),
                nS=int(info["nS"]),
                nC=int(info["nC"]),
                eigvals=np.asarray(info["eigvals"]),
                is_symmetric=bool(is_diagonal_symmetric or has_swapped_partner),
            )
        )
    return candidates


#Расстояние до равновесия
def log_distance_curve(traj: np.ndarray, eq: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(traj - eq[None, :], axis=1) #Для каждого момента времени считается расстояние ||x(t)-x(eq)||
    return np.log10(diff + EPS_LOG) #Возвращаем логарифм расстояния


def split_distance_curves(traj: np.ndarray, eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle_dist = np.linalg.norm(traj[:, [0, 2]] - eq[None, [0, 2]], axis=1) #Расстояние только по углам:sqrt((fi1-fi1(eq))^2+(fi2-fi2(eq)^2)
    vel_dist = np.linalg.norm(traj[:, [1, 3]] - eq[None, [1, 3]], axis=1) #Расстояние только по скоростям
    return np.log10(angle_dist + EPS_LOG), np.log10(vel_dist + EPS_LOG) #Возвращаются две кривые



#Выбор ближайшего равновесия
#Эта функция отвечает на вопрос: к какому равновесию траектория подошла ближе всего?
def pick_best_equilibrium(
    traj: np.ndarray,
    t: np.ndarray,
    candidates: Sequence[EquilibriumCandidate],
    source_eq: np.ndarray,
    prefer_symmetric: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, float, float, int]:
    if not candidates: #Если равновесий нет, анализ невозможен
        raise RuntimeError("Не удалось найти равновесия-кандидаты")

    working = list(candidates) #Копия списка кандидатов
    if prefer_symmetric: #Если пользователь хочет рассматривать только симметричные равновесия
        sym_only = [c for c in working if c.is_symmetric] #Отбираем симметричные
        if sym_only: #Если такие есть, дальше работаем только с ними
            working = sym_only

    # Переменные для лучшего найденного равновесия
    best_eq = None #координаты
    best_meta = None #информация
    best_curve = None #log10(||x(t) - best_eq||)
    best_min_val = None #минимальное значение расстояния (в логарифме)
    best_argmin = None #индекс времени, где достигается минимум

    cutoff = int(0.1 * len(traj))  # игнорируем первые 10%

    for c in working: #Перебираем равновесия
        if np.linalg.norm(c.point - source_eq) < 1e-6: #исключаем исходное равновесие
            continue

        # полная кривая нужна для графика
        curve_full = log_distance_curve(traj, c.point)
        # обрезанная кривая нужна только для выбора минимума
        curve_cut = curve_full[cutoff:]

        idx_local = int(np.argmin(curve_cut))
        val = float(curve_cut[idx_local])
        idx = idx_local + cutoff

        if best_min_val is None or val < best_min_val: #Если это равновесие ближе предыдущих — запоминаем его
            best_eq = c.point
            best_meta = { #Сохраняем информацию о типе равновесия
                "nU": c.nU,
                "nS": c.nS,
                "nC": c.nC,
                "eigvals": c.eigvals,
                "is_symmetric": c.is_symmetric,
            }
            best_curve = curve_full
            best_min_val = val
            best_argmin = idx

    if best_eq is None or best_meta is None or best_curve is None or best_argmin is None:
        best_eq = source_eq
        best_meta = {
            "nU": -1,
            "nS": -1,
            "nC": -1,
            "eigvals": np.array([]),
            "is_symmetric": False,
        }
        best_curve = log_distance_curve(traj, source_eq)
        best_min_val = float(np.min(best_curve))
        best_argmin = int(np.argmin(best_curve))

    return best_eq, best_meta, best_curve, float(best_min_val), float(t[best_argmin]), int(best_argmin)

def pick_common_separator_equilibrium(
    probes: Sequence[ProbeResult],
    candidates: Sequence[EquilibriumCandidate],
    cutoff_fraction: float = 0.1,
    threshold: float = -1.0,  # насколько "близко" должны пройти обе траектории (траектория должна подойти к равновесию ближе, чем на 0.1)
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:

    best_eq = None
    best_meta = None
    best_score = None

    for c in candidates:

        # не считаем разделителем равновесия,
        # из которых сами сепаратрисы стартовали
        is_source_eq = any(
            np.linalg.norm(c.point - p.source_eq) < 1e-5
            for p in probes
        )

        if is_source_eq:
            continue

        mins = []

        for p in probes:
            cutoff = int(cutoff_fraction * len(p.trajectory))
            traj_cut = p.trajectory[cutoff:]

            if len(traj_cut) == 0:
                continue

            curve = log_distance_curve(traj_cut, c.point)
            mins.append(float(np.min(curve)))

        if len(mins) != len(probes):
            continue

        # критерий: обе траектории должны быть близко
        score = max(mins)

        if best_score is None or score < best_score:
            best_score = score
            best_eq = c.point
            best_meta = {
                "nU": c.nU,
                "nS": c.nS,
                "nC": c.nC,
                "eigvals": c.eigvals,
                "is_symmetric": c.is_symmetric,
                "score": best_score,
                "mins": mins,
            }

    if best_eq is None or best_score is None:
        return None, None
    if best_score > threshold:
        # слишком далеко, значит это не "разделяющее" равновесие
        return None, None
    return np.asarray(best_eq, dtype=float), best_meta

def collect_probe_candidate_equilibria(
    probes: Sequence[ProbeResult],
    tol: float = 1e-8,
) -> List[EquilibriumCandidate]:
    collected: List[EquilibriumCandidate] = []

    for p in probes:
        for c in p.candidate_eqs:
            already_exists = any(
                np.linalg.norm(c.point - old.point) < tol
                for old in collected
            )

            if not already_exists:
                collected.append(c)

    return collected

#Эта функция переводит координаты точки на карте в параметры системы
def _build_params_for_probe(config: Dict[str, Any], x_probe: float, y_probe: float) -> Tuple[float, float, float]:
    def_sys = config["defaultSystem"] #Берём базовые параметры
    gamma = float(def_sys["gamma"]) #Начальные значения
    lam = float(def_sys["lambda"])
    k = float(def_sys["k"])

    param_x_name = config["grid"]["first"]["name"] #Узнаём, какие параметры меняются по осям
    param_y_name = config["grid"]["second"]["name"]

    params = {"gamma": gamma, "lambda": lam, "k": k} #Создаём словарь параметров
    params[param_x_name] = float(x_probe) #Подставляем значения конкретной точки карты
    params[param_y_name] = float(y_probe)
    return float(params["gamma"]), float(params["lambda"]), float(params["k"]) #Возвращаем полный набор параметров


#Эта функция выбирает две точки около границы
def _probe_positions(x_left: float, x_right: float, closeness: float = 0.0) -> Dict[str, float]:
    return {
        "left": float(x_left),
        "right": float(x_right),
    }


#Анализ одной пробной точки
#Эта функция делает всё для одной точки: строит сепаратрисное начальное условие, интегрирует траекторию и ищет ближайшее равновесие
def _make_probe_result(
    label: str,
    side: str,
    config: Dict[str, Any],
    x_probe: float,
    y_probe: float,
    dt_traj: float,
    n_steps_traj: int,
    stride_traj: int,
    prefer_symmetric: bool,
    prev_eq: Optional[np.ndarray] = None,
    prev_dir: Optional[np.ndarray] = None,
) -> ProbeResult:
    gamma, lam, k = _build_params_for_probe(config, x_probe, y_probe) #Получаем параметры системы
    candidate_eqs = find_candidate_equilibria(gamma=gamma, lam=lam, k=k)
    sep_cfg = config.get("separatrix_init", {}) #Берём настройки построения сепаратрисы

    source_eq, init_point, unstable_dir, branch_id = build_separatrix_init_for_point( #Строится начальная точка около неустойчивого равновесия
        gamma=gamma,
        lam=lam,
        k=k,
        saddle_focus_rule=sep_cfg.get("saddle_focus_rule", "phi1_lt_phi2"), #правило выбора седло-фокуса
        branch_rule=sep_cfg.get("branch_rule", "phi1_above_eq"), #правило выбора ветви
        offset_index=int(sep_cfg.get("offset_index", 1)), #какое направление брать
        eps_shift=float(sep_cfg.get("eps_shift", 1e-6)), #насколько далеко сдвинуться от равновесия
        dt_sep=float(sep_cfg.get("dt_sep", 1e-3)), #маленькое предварительное продвижение по траектории
        steps_sep=int(sep_cfg.get("steps_sep", 1)),
        prev_eq=prev_eq, #нужны, чтобы при переходе от точки к точке выбирать согласованную ветвь, а не случайно перескочить на другую
        ref_unstable_dir=prev_dir,
    )

    t, traj = integrate_trajectory( #Интегрируем траекторию из init_point
        y0=init_point,
        gamma=gamma,
        lam=lam,
        k=k,
        dt=dt_traj,
        n_steps=n_steps_traj,
        stride=stride_traj,
    )

    best_eq, best_meta, best_curve, best_min_val, best_t, best_idx = pick_best_equilibrium( #Ищем, к какому равновесию траектория подошла ближе всего
        traj=traj,
        t=t,
        candidates=candidate_eqs,
        source_eq=source_eq,
        prefer_symmetric=prefer_symmetric,
    )

    return ProbeResult( #Упаковываем всё в объект ProbeResult
        label=label,
        side=side,
        x_param=float(x_probe),
        y_param=float(y_probe),
        params=(gamma, lam, k),
        source_eq=np.asarray(source_eq, dtype=float),
        init_point=np.asarray(init_point, dtype=float),
        unstable_dir=np.asarray(unstable_dir, dtype=float),
        branch_id=int(branch_id),
        trajectory=np.asarray(traj, dtype=float),
        time=np.asarray(t, dtype=float),
        closest_eq=np.asarray(best_eq, dtype=float),
        closest_eq_meta=best_meta,
        log_distance_to_best_eq=np.asarray(best_curve, dtype=float),
        best_eq_min_log10=float(best_min_val),
        best_eq_argmin_t=float(best_t),
        best_eq_argmin_index=int(best_idx),
        candidate_eqs=list(candidate_eqs),
    )



#запускает полный анализ перехода
def analyze_separatrix_transition_v2(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    output_dir: str,
    row_index: Optional[int] = None,
    transition_number: int = 0,
    closeness: float = 0.5,
    dt_traj: Optional[float] = None,
    n_steps_traj: int = 30000,
    stride_traj: int = 1,
    prefer_symmetric_equilibria: bool = False,
) -> TransitionAnalysisResult:

    os.makedirs(output_dir, exist_ok=True) #Создаёт папку для результатов

    scan = extract_horizontal_scan(kneading_map_flat, config, row_index=row_index) #Берёт горизонтальную строку карты
    transitions = find_code_transitions_on_scan(scan) #Ищет все места, где меняется код
    if not transitions: #Если переходов нет — анализ невозможен
        raise RuntimeError("На выбранной горизонтальной линии не найден переход цвета")
    if transition_number < 0 or transition_number >= len(transitions): #Проверяет, что пользователь выбрал существующий номер перехода
        raise ValueError(f"transition_number={transition_number} вне диапазона [0, {len(transitions) - 1}]")

    left_idx, right_idx = transitions[transition_number] #Берётся нужный переход
    left_scan = scan[left_idx] #Получаем точки слева и справа от границы
    right_scan = scan[right_idx]

    x_boundary_est = 0.5 * (left_scan.x + right_scan.x) #Граница приближённо считается посередине
    y_scan = left_scan.y #Вертикальный параметр постоянен, потому что идём по горизонтальной строке
    probes_x = _probe_positions(left_scan.x, right_scan.x, closeness=closeness) #Получаем три точки: left, center, right.

    if dt_traj is None: #Если шаг интегрирования не задан, берём его из конфига
        dt_traj = float(config["kneadings_pendulums"]["dt"])

    gamma0, lam0, k0 = _build_params_for_probe(config, x_boundary_est, y_scan) #Берём параметры в центральной точке
    candidate_eqs = find_candidate_equilibria(gamma=gamma0, lam=lam0, k=k0) #Ищем равновесия около границы
    if not candidate_eqs: #Если равновесий нет — ошибка
        raise RuntimeError("Не удалось найти равновесия-кандидаты вблизи выбранной границы")

    probes: List[ProbeResult] = [] #Сюда будут складываться результаты для двух точек
    order = [("before", "left"), ("after", "right")] #Задаётся порядок анализа: до границы, после границы
    prev_eq = None #Сначала предыдущего равновесия и направления нет
    prev_dir = None
    for label, side in order: #Цикл по двум точкам
        pr = _make_probe_result(
            label=label,
            side=side,
            config=config,
            x_probe=probes_x[side],
            y_probe=y_scan,
            dt_traj=dt_traj,
            n_steps_traj=n_steps_traj,
            stride_traj=stride_traj,
            prefer_symmetric=prefer_symmetric_equilibria,
            prev_eq=prev_eq,
            prev_dir=prev_dir,
        )
        probes.append(pr) #Сохраняем результат
        prev_eq = pr.source_eq #Запоминаем равновесие и направление, чтобы следующая точка брала согласованную сепаратрису
        prev_dir = pr.unstable_dir

    common_candidates = collect_probe_candidate_equilibria(probes)

    common_eq, common_meta = pick_common_separator_equilibrium(
        probes=probes,
        candidates=common_candidates,
    )

    result = TransitionAnalysisResult( #Собираем общий результат
        row_index=int(left_scan.j),
        transition_left_index=int(left_idx),
        transition_right_index=int(right_idx),
        left_scan_point=left_scan,
        right_scan_point=right_scan,
        common_eq=common_eq,
        common_eq_meta=common_meta,
        x_boundary_estimate=float(x_boundary_est),
        y_scan_value=float(y_scan),
        probes=probes,
        candidate_equilibria=[
            {
                "point": np.asarray(c.point, dtype=float),
                "nU": int(c.nU),
                "nS": int(c.nS),
                "nC": int(c.nC),
                "eigvals": np.asarray(c.eigvals),
                "is_symmetric": bool(c.is_symmetric),
            }
            for c in common_candidates
        ],
        output_dir=output_dir,
    )

    save_transition_report(result) #Сохраняем .json-отчёт
    make_transition_plots_v2(config, kneading_map_flat, result) #Строим картинки
    return result


#Эти координаты полезны, потому что симметричное состояние имеет fi_=0 и v_=0
def _phi_pm_v_pm(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_plus = 0.5 * (traj[:, 0] + traj[:, 2]) #fi+=(fi1+fi2)/2
    phi_minus = 0.5 * (traj[:, 0] - traj[:, 2])
    v_plus = 0.5 * (traj[:, 1] + traj[:, 3])
    v_minus = 0.5 * (traj[:, 1] - traj[:, 3])
    return phi_plus, phi_minus, v_plus, v_minus


#Эта функция нужна, потому что JSON не понимает numpy-объекты напрямую
def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict): #Если это словарь — рекурсивно обрабатываем значения
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): #Если список или кортеж — обрабатываем элементы
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): #Массив numpy превращаем в обычный список
        return _to_jsonable(x.tolist())
    if isinstance(x, (np.floating,)): #np.float64 превращаем в обычный float
        return float(x)
    if isinstance(x, (np.integer,)): #np.int64 превращаем в обычный int
        return int(x)
    if isinstance(x, (np.bool_,)): #numpy boolean превращаем в обычный bool
        return bool(x)
    if isinstance(x, complex): #Комплексное число сохраняем как словарь
        return {"real": float(x.real), "imag": float(x.imag)}
    if isinstance(x, np.complexfloating): #То же самое для numpy-комплексных чисел
        return {"real": float(np.real(x)), "imag": float(np.imag(x))}
    return x



#Сохранение отчёта
def save_transition_report(result: TransitionAnalysisResult) -> None:
    data = { #Создаётся большой словарь с результатами (где найден переход; какие точки слева/справа; равновесия-кандидаты; параметры каждой траектории;closest_eq;минимум расстояния.)
        "row_index": result.row_index,
        "transition_left_index": result.transition_left_index,
        "transition_right_index": result.transition_right_index,
        "x_boundary_estimate": result.x_boundary_estimate,
        "y_scan_value": result.y_scan_value,
        "left_scan_point": asdict(result.left_scan_point),
        "right_scan_point": asdict(result.right_scan_point),
        "candidate_equilibria": result.candidate_equilibria,
        "common_eq": result.common_eq,
        "common_eq_meta": result.common_eq_meta,
        "probes": [
            {
                "label": p.label,
                "side": p.side,
                "x_param": p.x_param,
                "y_param": p.y_param,
                "params": p.params,
                "source_eq": p.source_eq,
                "init_point": p.init_point,
                "unstable_dir": p.unstable_dir,
                "branch_id": p.branch_id,
                "closest_eq": p.closest_eq,
                "closest_eq_meta": p.closest_eq_meta,
                "best_eq_min_log10": p.best_eq_min_log10,
                "best_eq_argmin_t": p.best_eq_argmin_t,
                "best_eq_argmin_index": p.best_eq_argmin_index,
            }
            for p in result.probes
        ],
    }
    with open(os.path.join(result.output_dir, "transition_report.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)


#Построение графиков, эта функция строит все картинки
def make_transition_plots_v2(
    config: Dict[str, Any],
    kneading_map_flat: np.ndarray,
    result: TransitionAnalysisResult,
) -> None:
    params_x, params_y, cols, rows = build_parameter_grid_from_config(config) #Готовим карту и горизонтальный срез
    arr2d = reshape_map(kneading_map_flat, cols, rows)
    scan = extract_horizontal_scan(kneading_map_flat, config, row_index=result.row_index)

    x_scan = np.array([p.x for p in scan], dtype=float) #Массивы для графика горизонтального среза
    raw_scan = np.array([p.raw_value for p in scan], dtype=float)

    x_min = float(np.min(params_x)) #Границы карты в реальных параметрах
    x_max = float(np.max(params_x))
    y_min = float(np.min(params_y))
    y_max = float(np.max(params_y))



    # 1. Карта параметров с реальными осями
    plt.figure(figsize=(8, 6))
    plt.imshow(arr2d, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max])
    plt.axhline(result.y_scan_value, linestyle="--")
    plt.scatter(
        [result.x_boundary_estimate],
        [result.y_scan_value],
        s=100,
        marker="x",
        color="red",
        label="boundary estimate",
    )
    plt.title("Карта параметров и выбранный переход")
    plt.xlabel(config["grid"]["first"].get("caption", config["grid"]["first"]["name"]))
    plt.ylabel(config["grid"]["second"].get("caption", config["grid"]["second"]["name"]))
    plt.legend()
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(result.output_dir, "01_parameter_map_real_axes.png"), dpi=250)
    plt.close()




    # 2. Горизонтальный срез
    plt.figure(figsize=(11, 4))

    plt.plot(
        x_scan,
        raw_scan,
        color="black",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="kneading value"
    )


    # оценка границы перехода
    plt.axvline(
        result.x_boundary_estimate,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="color/code change"
    )

    plt.title("Горизонтальный срез карты нидингов")
    plt.xlabel(config["grid"]["first"].get("caption", config["grid"]["first"]["name"]))
    plt.ylabel("raw kneading value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result.output_dir, "02_horizontal_scan.png"), dpi=250)
    plt.close()





    # 3. Набор 4 проекций 4D-траекторий
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # сколько времени рисуем на графике
    # если траектория всё равно улетает — поставь 10.0
    # если слишком коротко — поставь 100.0
    plot_t_max = 50.0

    projections = [
        ((0, 2), r"$(\phi_1, \phi_2)$", r"$\phi_1$", r"$\phi_2$"),
        ((1, 3), r"$(v_1, v_2)$", r"$v_1$", r"$v_2$"),
        ((0, 1), r"$(\phi_1, v_1)$", r"$\phi_1$", r"$v_1$"),
        ((2, 3), r"$(\phi_2, v_2)$", r"$\phi_2$", r"$v_2$"),
    ]

    probe_colors = {
        "before": "tab:orange",
        "after": "tab:green",
    }

    for ax, ((i, j), title, xlabel, ylabel) in zip(axes.ravel(), projections):

        # равновесия-кандидаты
        for eqi, item in enumerate(result.candidate_equilibria):
            eq = np.asarray(item["point"], dtype=float)


            if item["is_symmetric"]:
                ax.scatter(
                    eq[i], eq[j],
                    s=80,
                    marker="*",
                    color="gray",
                    alpha=0.45,
                    label="symmetric eq" if eqi == 0 else None,
                    zorder=2,
                )
            else:
                ax.scatter(
                    eq[i], eq[j],
                    s=35,
                    marker="o",
                    color="gray",
                    alpha=0.35,
                    label="candidate eq" if eqi == 0 else None,
                    zorder=1,
                )

        # Общее равновесие, которое "ломает" сепаратрисы
        if result.common_eq is not None:
            eq = result.common_eq

            ax.scatter(
                eq[i],
                eq[j],
                color="red",
                marker="*",
                s=180,
                edgecolor="black",
                linewidth=0.7,
                label="separator eq",
                zorder=10,
            )

        # траектории before / after
        for p in result.probes:
            if p.label == "near":
                continue

            color = probe_colors.get(p.label, "black")

            # ограничиваем траекторию по времени
            mask = p.time <= plot_t_max
            traj_plot = p.trajectory[mask]
            time_plot = p.time[mask]

            # если вдруг mask оказался пустым, берём хотя бы первую точку
            if len(traj_plot) == 0:
                traj_plot = p.trajectory[:1]
                time_plot = p.time[:1]

            ax.plot(
                traj_plot[:, i],
                traj_plot[:, j],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=p.label,
                zorder=5,
            )

            # начальная точка отображаемого куска
            ax.scatter(
                traj_plot[0, i],
                traj_plot[0, j],
                color=color,
                marker="o",
                s=35,
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )

            # конечная точка отображаемого куска
            ax.scatter(
                traj_plot[-1, i],
                traj_plot[-1, j],
                color=color,
                marker="x",
                s=60,
                linewidth=2,
                zorder=7,
            )

            # равновесие, из которого выпускали сепаратрису
            ax.scatter(
                p.source_eq[i],
                p.source_eq[j],
                color=color,
                marker="D",
                s=45,
                edgecolor="black",
                linewidth=0.5,
                zorder=8,
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.margins(0.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        f"Сепаратрисы в 2D-проекциях 4D-фазового пространства, t ≤ {plot_t_max}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(os.path.join(result.output_dir, "03_phase_projections_4d.png"), dpi=250)
    plt.close()





    # 4. Расстояние до организующего равновесия
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    plot_t_max = 50.0

    probe_colors = {
        "before": "tab:orange",
        "after": "tab:green",
    }

    # Если найдено общее равновесие-разделитель, считаем расстояние до него.
    # Если нет — для каждой траектории считаем расстояние до её closest_eq.
    use_common_eq = result.common_eq is not None

    for p in result.probes:
        color = probe_colors.get(p.label, "black")

        mask = p.time <= plot_t_max
        if not np.any(mask):
            mask = np.ones_like(p.time, dtype=bool)

        t_plot = p.time[mask]
        traj_plot = p.trajectory[mask]

        if use_common_eq:
            eq = np.asarray(result.common_eq, dtype=float)
            eq_label = "common separator eq"
        else:
            eq = np.asarray(p.closest_eq, dtype=float)
            eq_label = "individual closest eq"

        full_curve = log_distance_curve(traj_plot, eq)
        angle_curve, vel_curve = split_distance_curves(traj_plot, eq)

        min_idx = int(np.argmin(full_curve))
        min_t = float(t_plot[min_idx])
        min_val = float(full_curve[min_idx])

        axes[0].plot(t_plot, full_curve, color=color, linewidth=2.0, label=p.label)
        axes[1].plot(t_plot, angle_curve, color=color, linewidth=2.0, label=p.label)
        axes[2].plot(t_plot, vel_curve, color=color, linewidth=2.0, label=p.label)

        # отмечаем минимум на верхнем графике
        axes[0].scatter(
            [min_t],
            [min_val],
            color=color,
            marker="o",
            s=55,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        axes[0].annotate(
            f"min={min_val:.2f}",
            xy=(min_t, min_val),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    # горизонтальные пороги близости
    for ax in axes:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axhline(-1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axhline(-2.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.4)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylabel("log10 distance")

    axes[0].set_title(
        rf"Расстояние до {eq_label}: $\log_{{10}}(\|x(t)-x_{{eq}}\|+10^{{-30}})$"
    )

    axes[1].set_title(
        r"Угловая часть расстояния: "
        r"$\log_{10}(\|(\phi_1,\phi_2)-(\phi_{1,eq},\phi_{2,eq})\|+10^{-30})$"
    )

    axes[2].set_title(
        r"Скоростная часть расстояния: "
        r"$\log_{10}(\|(v_1,v_2)-(v_{1,eq},v_{2,eq})\|+10^{-30})$"
    )

    axes[2].set_xlabel("t")

    fig.suptitle(
        f"Близость сепаратрис к равновесию, t ≤ {plot_t_max:g}",
        fontsize=14,
    )

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass

    plt.savefig(os.path.join(result.output_dir, "05_distance_to_candidate_equilibria.png"), dpi=250)
    plt.close()






    # 5. Временные ряды: wrapped/unwrapped
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)

    probe_colors = {
        "before": "orange",
        "after": "green",
    }

    label_map = {
        "before": "до границы",
        "after": "после границы",
    }

    for p in result.probes:
        phi1 = p.trajectory[:, 0]
        phi2 = p.trajectory[:, 2]

        color = probe_colors.get(p.label, "black")
        label = label_map.get(p.label, p.label)

        # unwrapped
        axes[0, 0].plot(p.time, phi1, color=color, linewidth=2.0, label=f"φ₁ {label}")
        axes[0, 1].plot(p.time, phi2, color=color, linewidth=2.0, label=f"φ₂ {label}")

        # wrapped
        axes[1, 0].plot(p.time, wrap_angle_0_2pi(phi1), color=color, linewidth=2.0, label=f"φ₁ mod 2π {label}")
        axes[1, 1].plot(p.time, wrap_angle_0_2pi(phi2), color=color, linewidth=2.0, label=f"φ₂ mod 2π {label}")

    # заголовки
    titles = [
        r"Развёрнутый угол $\phi_1(t)$",
        r"Развёрнутый угол $\phi_2(t)$",
        r"$\phi_1(t) \ \mathrm{mod}\ 2\pi$",
        r"$\phi_2(t) \ \mathrm{mod}\ 2\pi$",
    ]

    # оформление осей
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t")
        ax.set_ylabel("угол")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # общий заголовок
    fig.suptitle("Временные ряды углов маятников", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(result.output_dir, "06_time_series_wrapped_unwrapped.png"), dpi=250)
    plt.close()





    # 6. Отдельная фигура "до / после" в проекции (phi1, phi2) без mod 2pi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    t_max_plot = plot_t_max

    probe_colors = {
        "before": "tab:orange",
        "after": "tab:green",
    }

    label_map = {
        "before": "до границы",
        "after": "после границы",
    }

    for ax, p in zip(axes, result.probes):
        mask = p.time <= t_max_plot
        traj_plot = p.trajectory[mask]

        if len(traj_plot) == 0:
            traj_plot = p.trajectory[:1]

        color = probe_colors.get(p.label, "black")
        label = label_map.get(p.label, p.label)

        phi1 = traj_plot[:, 0]
        phi2 = traj_plot[:, 2]

        ax.plot(
            phi1,
            phi2,
            color=color,
            linewidth=2.0,
            label=f"сепаратриса {label}",
        )

        ax.scatter(
            phi1[0],
            phi2[0],
            s=70,
            marker="D",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label="начало траектории",
            zorder=5,
        )

        ax.scatter(
            phi1[-1],
            phi2[-1],
            s=80,
            marker="x",
            color=color,
            linewidth=2.0,
            label="конец траектории",
            zorder=6,
        )

        ax.scatter(
            p.source_eq[0],
            p.source_eq[2],
            s=90,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label="исходное равновесие",
            zorder=7,
        )

        ax.scatter(
            p.closest_eq[0],
            p.closest_eq[2],
            s=140,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=0.6,
            label="ближайшее равновесие",
            zorder=8,
        )

        ax.set_title(f"{label}, t ≤ {t_max_plot:g}")
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Проекция сепаратрис без свёртки углов")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(result.output_dir, "07_before_after_transition.png"), dpi=250)
    plt.close()


def make_human_summary(result: TransitionAnalysisResult) -> str:
    lines: List[str] = []
    lines.append("АНАЛИЗ ПЕРЕХОДА МЕЖДУ ДВУМЯ КОДАМИ НИДИНГА")
    lines.append("")
    lines.append(f"Горизонтальная линия: row_index={result.row_index}, y={result.y_scan_value:.15f}")
    lines.append(
        f"Переход найден между столбцами i={result.transition_left_index} и i={result.transition_right_index}"
    )
    lines.append(
        f"Слева код={result.left_scan_point.code}, x={result.left_scan_point.x:.15f}; "
        f"справа код={result.right_scan_point.code}, x={result.right_scan_point.x:.15f}"
    )
    lines.append(f"Оценка положения границы: x* ~ {result.x_boundary_estimate:.15f}")
    lines.append("")

    lines.append("КАНДИДАТЫ В ОРГАНИЗУЮЩИЕ РАВНОВЕСИЯ:")
    for idx, eq in enumerate(result.candidate_equilibria):
        point = np.asarray(eq["point"], dtype=float)
        lines.append(
            f"  eq[{idx}] = {np.array2string(point, precision=10)}, "
            f"nU={eq['nU']}, nS={eq['nS']}, nC={eq['nC']}, symmetric={eq['is_symmetric']}"
        )
    lines.append("")

    for p in result.probes:
        lines.append(f"[{p.label.upper()} / {p.side.upper()}]")
        lines.append(
            f"params = (gamma={p.params[0]:.15f}, lambda={p.params[1]:.15f}, k={p.params[2]:.15f})"
        )
        lines.append(f"source saddle-focus eq = {np.array2string(p.source_eq, precision=12)}")
        lines.append(f"init point            = {np.array2string(p.init_point, precision=12)}")
        lines.append(f"closest candidate eq  = {np.array2string(p.closest_eq, precision=12)}")
        lines.append(
            f"min log10 distance    = {p.best_eq_min_log10:.6f} at t={p.best_eq_argmin_t:.6f}"
        )
        lines.append(
            f"best eq meta          = symmetric={p.closest_eq_meta['is_symmetric']}, "
            f"nU={p.closest_eq_meta['nU']}, nS={p.closest_eq_meta['nS']}, nC={p.closest_eq_meta['nC']}"
        )
        lines.append("")

    lines.append("ИНТЕРПРЕТАЦИЯ:")
    lines.append(
        "Если траектории до, около и после границы качественно различаются, это указывает на перестройку сепаратрисы при проходе через границу нидингового кода."
    )
    lines.append(
        "Если график log10-расстояния имеет глубокий провал, это означает близкий подход к соответствующему равновесию-кандидату."
    )
    lines.append(
        "Для 4D-системы одной проекции (phi1, phi2) недостаточно, поэтому дополнительно строятся проекции со скоростями и симметричными координатами (phi+, phi-, v+, v-)."
    )
    return "\n".join(lines)


def save_human_summary(result: TransitionAnalysisResult) -> None:
    with open(os.path.join(result.output_dir, "transition_summary.txt"), "w", encoding="utf-8") as f:
        f.write(make_human_summary(result))


__all__ = [
    "analyze_separatrix_transition_v2",
    "extract_horizontal_scan",
    "find_code_transitions_on_scan",
    "find_candidate_equilibria",
    "make_human_summary",
    "save_human_summary",
]
