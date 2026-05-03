import math
import numpy as np
from scipy.optimize import root #численное решение систем нелинейных уравнений


# Правая часть системы
def pendulum_rhs(y, gamma, lam, k):
    fi1, v1, fi2, v2 = y
    return np.array([
        v1,
        -lam * v1 - math.sin(fi1) + gamma + k * math.sin(fi2 - fi1),
        v2,
        -lam * v2 - math.sin(fi2) + gamma + k * math.sin(fi1 - fi2)
    ], dtype=float)


def rk4_step(y, dt, gamma, lam, k):
    k1 = pendulum_rhs(y, gamma, lam, k)
    k2 = pendulum_rhs(y + 0.5 * dt * k1, gamma, lam, k)
    k3 = pendulum_rhs(y + 0.5 * dt * k2, gamma, lam, k)
    k4 = pendulum_rhs(y + dt * k3, gamma, lam, k)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Равновесия
def equilibrium_equations(x, gamma, k):
    fi1, fi2 = x
    return np.array([
        math.sin(fi1) - gamma - k * math.sin(fi2 - fi1),
        math.sin(fi2) - gamma - k * math.sin(fi1 - fi2)
    ], dtype=float)


def equilibrium_jacobian(x, gamma, k):
    fi1, fi2 = x
    c = math.cos(fi2 - fi1)
    return np.array([
        [math.cos(fi1) + k * c, -k * c],
        [-k * c, math.cos(fi2) + k * c]
    ], dtype=float)


#функция нормализации угла в диапазон [0,2pi)
def wrap_angle_0_2pi(phi):
    return phi % (2.0 * math.pi)


def wrap_equilibrium_point(eq):
    eq = np.array(eq, dtype=float).copy() #Преобразует входные данные в numpy-массив и создаёт копию (чтобы не изменять оригинал)
    eq[0] = wrap_angle_0_2pi(eq[0]) #нормализует fi1
    eq[2] = wrap_angle_0_2pi(eq[2]) #нормализует fi2
    eq[1] = 0.0 #обнуление v1
    eq[3] = 0.0 #обнуление v2
    return eq #возвращает нормализованную точку равновесия


#Эта функция пытается найти равновесие, начиная с начального предположения guess = [fi1, fi2]
def solve_equilibrium_from_guess(gamma, k, guess, tol=1e-12):
    sol = root(
        lambda x: equilibrium_equations(x, gamma, k), #Передаётся функция невязок. функция, которая фиксирует gamma и k, а меняет только x
        np.array(guess, dtype=float), #Это начальное приближение.
        jac=lambda x: equilibrium_jacobian(x, gamma, k),
        method="lm", #метод Левенберга-Марквардта (гибрид Ньютона и градиентного спуска)
        options={"xtol": tol}
    )
    if not sol.success: #не удалось найти корень
        return None
    fi1, fi2 = sol.x #Если всё хорошо — достаются найденные углы
    return wrap_equilibrium_point([fi1, 0.0, fi2, 0.0]) #Возвращает 4D-точку [fi1, 0, fi2, 0]


def _base_asin_gamma(gamma):
    if abs(gamma) <= 1.0:
        return math.asin(max(-1.0, min(1.0, gamma)))
    return 0.0



# Главная функция поиска всех равновесий системы. Возвращает список 4D-точек равновесий.
def find_equilibria_pendulum(gamma, k, tol=1e-12): #λ не влияет на положения равновесий, только на их устойчивость.
    a = _base_asin_gamma(gamma) #Вычисление базового угла a = arcsin(γ)
    guesses = [ #Четыре начальных приближения, соответствующие симметриям системы
        (a, a),
        (a, math.pi - a),
        (math.pi - a, a),
        (math.pi - a, math.pi - a),
    ]
    sols = [] #Список для хранения уникальных найденных равновесий
    for guess in guesses:
        sol = root( #Решение системы уравнений для текущего приближения
            lambda x: equilibrium_equations(x, gamma, k),
            np.array(guess, dtype=float),
            jac=lambda x: equilibrium_jacobian(x, gamma, k),
            method="lm",
            options={"xtol": tol}
        )
        if not sol.success: #Если решение не найдено — переходим к следующему приближению
            continue
        cand = wrap_equilibrium_point([sol.x[0], 0.0, sol.x[1], 0.0]) #Создаём 4D-точку равновесия и нормализуем углы

        is_new = True #Проверка на дубликат. Вычисляем евклидово расстояние между кандидатом и уже найденными равновесиями.
        for s in sols: #Если расстояние меньше 1e-8 — считаем, что это то же самое равновесие.
            if np.linalg.norm(cand - s) < 1e-8:
                is_new = False
                break

        if is_new: #Если равновесие уникально — добавляем в список
            sols.append(cand)

    return sols # Возвращаем список всех найденных равновесий



#Вычисляет матрицу Якоби 4×4 для полной системы (с учётом скоростей). Нужна для анализа устойчивости равновесий.
def full_jacobian(eq, gamma, lam, k):
    fi1, _, fi2, _ = eq
    c12 = math.cos(fi2 - fi1)
    c21 = math.cos(fi1 - fi2)

    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-math.cos(fi1) - k * c12, -lam,  k * c12, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [ k * c21, 0.0, -math.cos(fi2) - k * c21, -lam]
    ], dtype=float)


#Классифицирует тип равновесия на основе собственных значений якобиана
def equilibrium_type(eq, gamma, lam, k, eps=1e-10):
    J = full_jacobian(eq, gamma, lam, k) #Вычисление собственных значений и собственных векторов якобиана
    eigvals, eigvecs = np.linalg.eig(J)

    nU = sum(ev.real > eps for ev in eigvals) #число неустойчивых направлений (Re(λ) > 0)
    nS = sum(ev.real < -eps for ev in eigvals)  #число устойчивых направлений (Re(λ) < 0)
    nC = len(eigvals) - nU - nS #Если часть направлений устойчивые, часть неустойчивые, то оставшиеся — центральные: это те, у которых вещественная часть очень близка к нулю

    return { #Возвращает словарь с полной информацией о равновесии
        "point": np.array(eq, dtype=float), #само равновесие
        "eigvals": eigvals, #собственные значения
        "eigvecs": eigvecs, #собственные векторы
        "nU": int(nU), #размерности многообразий
        "nS": int(nS),
        "nC": int(nC),
    }


#Является ли данное равновесие с одномерным неустойчивым многообразием?
#То есть проверяет два условия: 1. Есть ровно одно неустойчивое направление 2. Есть комплексные собственные значения
def is_saddle_focus_1d_unstable(eq_info, eps=1e-10):
    eigvals = eq_info["eigvals"] #Из словаря eq_info достаются собственные значения якобиана
    nU = sum(ev.real > eps for ev in eigvals) #Считается число неустойчивых направлений
    has_complex = any(abs(ev.imag) > eps for ev in eigvals) #есть ли хотя бы одно комплексное собственное значение
    return (nU == 1) and has_complex

#выбирает одно конкретное равновесие из списка eq_infos (с-ф с 1D неустойчивым направлением)
# Если prev_eq задано, приоритет у ближайшего продолжения.
def choose_target_saddle_focus(eq_infos, saddle_focus_rule="phi1_lt_phi2", prev_eq=None):
    candidates = [info for info in eq_infos if is_saddle_focus_1d_unstable(info)] #Из всех равновесий остаются только те, которые имеют 1 неуст нарпавл и комплексные собств знач

    if not candidates: #Если таких нет — нет нужного типа равновесия -> дальше строить сепаратрису нельзя
        raise RuntimeError("No saddle-focus with 1D unstable manifold found")

    if prev_eq is not None: #Если есть предыдущее равновесие
        prev_eq = np.array(prev_eq, dtype=float) #Просто приводим к массиву
        candidates.sort(key=lambda info: np.linalg.norm(info["point"] - prev_eq)) #Сортируем кандидатов по расстоянию до предыдущего равновесия
        return candidates[0] #Берём ближайшее

    if saddle_focus_rule == "phi1_lt_phi2": #Если prev_eq не задан
        filtered = [info for info in candidates if info["point"][0] < info["point"][2]] #Берутся только те равновесия, где fi1<fi2
        if filtered: #Если такие есть — берём первое
            return filtered[0]
        return candidates[0] #Если нет — берём просто первое из всех кандидатов

    raise ValueError(f"Unknown saddle_focus_rule: {saddle_focus_rule}")


def find_target_saddle_focus_at_point(gamma, lam, k, saddle_focus_rule="phi1_lt_phi2", prev_eq=None):
    equilibria = find_equilibria_pendulum(gamma, k) #Здесь вызывается функция поиска равновесий. Она возвращает список точек вида [fi1, 0, fi2, 0]
    if not equilibria: #Если список пустой, выбрасывается ошибка
        raise RuntimeError("No equilibria found")

    #Для каждого найденного равновесия eq вызывается equilibrium_type(eq, gamma, lam, k)
    #Каждый элемент списка — это словарь с информацией о равновесии. То есть после этой строки у нас уже не просто координаты равновесий, а полная характеристика каждого равновесия.
    eq_infos = [equilibrium_type(eq, gamma, lam, k) for eq in equilibria]
    #Теперь из всех равновесий выбирается одно нужное. оставляет только те равновесия, которые являются с-ф c 1D неуст многообр
    #если задан prev_eq, выбирает ближайшее к нему. если prev_eq нет, использует правило saddle_focus_rule (fi1<fi2)
    return choose_target_saddle_focus(
        eq_infos,
        saddle_focus_rule=saddle_focus_rule,
        prev_eq=prev_eq
    )



#находит единичный собственный вектор, соответствующий неустойчивому направлению
def unstable_direction_from_eq_info(eq_info, ref_dir=None, eps=1e-10):
    eigvals = eq_info["eigvals"] #собственные значения
    eigvecs = eq_info["eigvecs"] #соответствующие собственные векторы

    unstable_vec = None #Пока не нашли нужный вектор

    for i, ev in enumerate(eigvals): #Перебираем все собственные значения (и их индексы)
        if ev.real > eps: #Ищем неустойчивое собственное значение:
            vec = eigvecs[:, i].real #Берём соответствующий собственный вектор.
            nrm = np.linalg.norm(vec) #Считаем длину вектора
            if nrm == 0.0: #Если вдруг вектор нулевой — пропускаем
                continue
            unstable_vec = vec / nrm #Нормируем
            break #Теперь это единичный вектор

    if unstable_vec is None: #Если не нашли — значит нет неустойчивого направления
        raise RuntimeError("No unstable eigenvector found")

    if ref_dir is not None: #Если задан опорный вектор (от соседней точки параметров)
        ref_dir = np.array(ref_dir, dtype=float) #Приводим к numpy-массиву
        nr = np.linalg.norm(ref_dir) #Считаем его длину
        if nr > 0.0:
            ref_dir = ref_dir / nr #Нормируем его
            #в одной точке получилось v, в другой -v
            #если положительное, то направление согласовано, если отрицательное, то "переворачиваем"
            if np.dot(unstable_vec, ref_dir) < 0.0:
                unstable_vec = -unstable_vec

    return unstable_vec #единичный вектор неустойчивого направления



# Локальная сепаратриса и выбор ее ветви
# Строит короткую траекторию (кусок сепаратрисы), начиная очень близко к равновесию
def integrate_local_separatrix(eq, start_point, gamma, lam, k, dt_sep, steps_sep):
    traj = [np.array(start_point, dtype=float)] #Создаётся список траектории. первая точка = start_point это уже точка, немного сдвинутая от равновесия вдоль unstable направления
    y = np.array(start_point, dtype=float) #Текущая точка состояния системы

    for _ in range(steps_sep): #Цикл на steps_sep шагов интегрирования
        y = rk4_step(y, dt_sep, gamma, lam, k) #Делаем один шаг вперёд по времени методом Рунге–Кутты 4-го порядка
        traj.append(y.copy()) #Добавляем новую точку в траекторию

    return np.array(traj, dtype=float) #Возвращаем всю траекторию как массив


#У 1D неустойчивого направления есть две ветви. Нужно выбрать одну
def choose_separatrix_branch(eq, p_plus, p_minus, branch_rule="phi1_above_eq"):
    if branch_rule == "phi1_above_eq":
        if p_plus[0] > eq[0]: #Смотрим на первую координату (fi1). если у точки p_plus угол больше, чем у равновесия, то выбираем ветвь +
            return +1, np.array(p_plus, dtype=float) #+1 — идентификатор ветви. возвращаем p_plus
        return -1, np.array(p_minus, dtype=float) #берём противоположную ветвь

    raise ValueError(f"Unknown branch_rule: {branch_rule}") #Если правило неизвестно — ошибка


#начальное условие для запуска траектории по неустойчивой сепаратрисе
def build_init_from_equilibrium(
    eq, #Точка равновесия
    gamma,
    lam,
    k,
    branch_rule="phi1_above_eq", #Правило выбора ветви сепаратрисы
    offset_index=1, #Номер точки на короткой траектории, которую мы возьмём как init_point
    eps_shift=1e-6, #Очень маленький сдвиг от равновесия вдоль неустойчивого направления
    dt_sep=1e-3, #Шаг по времени при построении локальной сепаратрисы
    steps_sep=1, #Сколько шагов пройти вдоль сепаратрисы
    ref_unstable_dir=None, #Опорное направление для согласования знака собственного вектора
):
    info = equilibrium_type(eq, gamma, lam, k) #Здесь вычисляется полная линейная информация о равновесии (якобиан, сч, св, число устойч и неустойч направл)

    #подходит ли это равновесие для построения 1D неустойчивой сепаратрисы?
    if not is_saddle_focus_1d_unstable(info):
        raise RuntimeError("Equilibrium is not a saddle-focus with 1D unstable manifold")

    unstable_dir = unstable_direction_from_eq_info(info, ref_dir=ref_unstable_dir) #Здесь находится единичный вектор unstable_dir, соответствующий неустойчивому собственному значению

    #Из равновесия строятся две близкие точки
    p_plus = eq + eps_shift * unstable_dir #начальные точки на двух ветвях
    p_minus = eq - eps_shift * unstable_dir

    branch_id, start_point = choose_separatrix_branch( #Здесь выбирается одна из двух ветвей
        eq=eq,
        p_plus=p_plus,
        p_minus=p_minus,
        branch_rule=branch_rule,
    )

    traj = integrate_local_separatrix( #Теперь строится короткая траектория, начиная из start_point. В итоге traj — это массив точек траектории.
        eq=eq,
        start_point=start_point,
        gamma=gamma,
        lam=lam,
        k=k,
        dt_sep=dt_sep,
        steps_sep=steps_sep,
    )

    if offset_index < 0 or offset_index >= len(traj): #Проверка допустимости индекса
        raise ValueError(
            f"offset_index={offset_index} out of range, trajectory length={len(traj)}"
        )

    init_point = traj[offset_index].copy() #Вот здесь выбирается итоговая начальная точка (start_point находится слишком близко к равновесию, поэтому его нельзя брать)

    return {
        "eq_point": np.array(eq, dtype=float), #равновесие
        "init_point": np.array(init_point, dtype=float), #точка на локальной сепаратрисе, которую потом можно использовать как начальное условие
        "eq_info": info, #Полная информация о типе равновесия
        "unstable_dir": unstable_dir, #Единичный вектор неустойчивого направления
        "branch_id": int(branch_id), #Номер выбранной ветви +1 или -1
        "local_traj": traj, #Весь локальный кусок сепаратрисы, а не только одна точка
    }


#конкретная точка, с которой надо стартовать, чтобы попасть на нужную неустойчивую сепаратрису
def build_separatrix_init_for_point(
    gamma,
    lam,
    k,
    saddle_focus_rule="phi1_lt_phi2", #выбирать равновесие, где fi1<fi2
    branch_rule="phi1_above_eq",
    offset_index=1,
    eps_shift=1e-6,
    dt_sep=1e-3,
    steps_sep=1,
    prev_eq=None, #Если задано предыдущее равновесие, то новое выбирается как ближайшее к нему
    ref_unstable_dir=None,
):
    target_eq_info = find_target_saddle_focus_at_point( #находит все равновесия,для каждого определяет тип, оставляет только с-ф с 1D unstable, выбирает одно конкретное равновесие
        gamma=gamma,
        lam=lam,
        k=k,
        saddle_focus_rule=saddle_focus_rule,
        prev_eq=prev_eq,
    )

    result = build_init_from_equilibrium( #ещё раз проверяет тип равновесия, находит неустойчивый собственный вектор, строит две точки, выбирает нужную ветвь, интегрирует короткую траекторию вдоль этой ветви, берёт точку с номером offset_index
        eq=target_eq_info["point"],
        gamma=gamma,
        lam=lam,
        k=k,
        branch_rule=branch_rule,
        offset_index=offset_index,
        eps_shift=eps_shift,
        dt_sep=dt_sep,
        steps_sep=steps_sep,
        ref_unstable_dir=ref_unstable_dir,
    )

    return (
        result["eq_point"], #найденное равновесие
        result["init_point"], #Точка на выбранной ветви неустойчивой сепаратрисы, которую потом можно использовать как начальное условие для дальнейшего интегрирования
        result["unstable_dir"],#Единичный вектор неустойчивого направления
        result["branch_id"], #Идентификатор ветви +1/ -1
    )



# Продолжение равновесия по сетке
def continue_target_equilibrium_on_grid(
    params_x,
    params_y,
    def_params,
    param_x_name,
    param_y_name,
    param_to_index,
    cols,
    rows,
    center_i,
    center_j,
    saddle_focus_rule="phi1_lt_phi2",
):
    total = len(params_x) #Общее число точек сетки
    eq_grid = [None] * total #Создаётся массив результатов длины total

    center_idx = center_i + center_j * cols #Выбор центральной точки
    center_params = np.array(def_params, dtype=float) #Создаётся копия базового вектора параметров
    center_params[param_to_index[param_x_name]] = params_x[center_idx] #В эту копию подставляются значения параметров, соответствующие центральной точке сетки
    center_params[param_to_index[param_y_name]] = params_y[center_idx]

    #Из вектора параметров извлекаются три конкретных значения
    gamma0 = float(center_params[param_to_index["gamma"]])
    lam0   = float(center_params[param_to_index["lambda"]])
    k0     = float(center_params[param_to_index["k"]])

    #Здесь в центральной точке параметров находится нужное равновесие
    center_eq_info = find_target_saddle_focus_at_point(
        gamma=gamma0,
        lam=lam0,
        k=k0,
        saddle_focus_rule=saddle_focus_rule,
        prev_eq=None, #Потому что это первый шаг — продолжать пока не от чего
    )
    eq_grid[center_idx] = center_eq_info["point"] #Найденная точка равновесия сохраняется в сетке

    deltas = [] #Список смещений от центральной точки
    #Здесь строятся все относительные координаты (di,dj) всех точек сетки относительно центра
    for dj in range(-center_j, rows - center_j):
        for di in range(-center_i, cols - center_i):
            if di == 0 and dj == 0:
                continue
            deltas.append((di, dj))
    deltas.sort(key=lambda d: abs(d[0]) + abs(d[1])) #сначала обрабатываются ближайшие точки, потом более далёкие

    #Перебираем все точки, кроме центра, в порядке возрастания расстояния от центра
    for di, dj in deltas:
        i = center_i + di #Переходим от относительных координат обратно к абсолютным индексам сетки
        j = center_j + dj
        idx = i + j * cols

        params = np.array(def_params, dtype=float) #Создаётся копия базовых параметров
        params[param_to_index[param_x_name]] = params_x[idx] #Подставляются значения параметров для текущего узла сетки
        params[param_to_index[param_y_name]] = params_y[idx]

        gamma = float(params[param_to_index["gamma"]]) #Извлекаются значения gamma, lam, k для текущей точки
        lam   = float(params[param_to_index["lambda"]])
        k     = float(params[param_to_index["k"]])

        neighbor_indices = [] #Список индексов соседей
        if i - 1 >= 0:
            neighbor_indices.append((i - 1) + j * cols) #Добавляется левый сосед, если он существует
        if i + 1 < cols:
            neighbor_indices.append((i + 1) + j * cols) #Добавляется правый сосед
        if j - 1 >= 0:
            neighbor_indices.append(i + (j - 1) * cols) #Добавляется сосед снизу по одной из осей
        if j + 1 < rows:
            neighbor_indices.append(i + (j + 1) * cols) #Добавляется противоположный вертикальный сосед

        guesses = [] #Список начальных приближений
        #Для каждого соседа: сли у него уже найдено равновесие,берутся только углы fi1, fi2 и добавляются в guesses
        for nidx in neighbor_indices:
            if eq_grid[nidx] is not None:
                guesses.append(eq_grid[nidx][[0, 2]])

        found_eq = None #Переменная для будущего найденного равновесия

        #Сначала не делается глобальный поиск, а пробуется “протянуть” равновесие из соседних точек
        for guess in guesses:
            cand = solve_equilibrium_from_guess(gamma, k, guess) #Из соседнего initial guess запускается решение системы уравнений равновесия (То есть предполагается: в близкой точке параметров нужное равновесие изменилось не сильно, значит соседнее равновесие — хороший старт.)
            if cand is None: #Если численно не сошлось — пробуем следующий guess
                continue

            info = equilibrium_type(cand, gamma, lam, k) #Если кандидат найден, определяется его тип
            if not is_saddle_focus_1d_unstable(info): #Если это не нужный тип равновесия, кандидат отбрасывается
                continue

            if saddle_focus_rule == "phi1_lt_phi2": #Если используется правило выбора "phi1_lt_phi2", то дополнительно проверяется fi1<fi2
                if info["point"][0] < info["point"][2]:
                    found_eq = info["point"]
                    break #Если условие выполнено — кандидат принимается
            else: #Если правило другое, то кандидат принимается без этой дополнительной проверки
                found_eq = info["point"]
                break

        # Fallback глобальный поиск. Если continuation от соседей не сработал, тогда включается запасной план.
        if found_eq is None:
            try:
                prev_eq = None
                if len(guesses) > 0: #Если хотя бы один guess был, из него строится полная 4D-точка
                    prev_eq = np.array([guesses[0][0], 0.0, guesses[0][1], 0.0], dtype=float)

                #Теперь вызывается более общий и дорогой поиск: находятся все равновесия,классифицируются, выбирается нужный saddle-focus, если prev_eq задан, берётся ближайший к нему.
                target_eq_info = find_target_saddle_focus_at_point(
                    gamma=gamma,
                    lam=lam,
                    k=k,
                    saddle_focus_rule=saddle_focus_rule,
                    prev_eq=prev_eq,
                )
                found_eq = target_eq_info["point"] #Из найденной информации берётся сама точка равновесия

            except Exception:  #Если всё рухнуло, просто сохраняется None
                found_eq = None

        eq_grid[idx] = found_eq #сохранение результата

    return eq_grid #Возвращается весь список равновесий по сетке.



# для каждой точки сетки, где равновесие найдено, строит начальное условие init_point на нужной ветви локальной неустойчивой сепаратрисы
def build_inits_from_eq_grid(
    eq_grid, #сетка равновесий
    params_x,
    params_y,
    def_params,
    param_x_name,
    param_y_name,
    param_to_index,
    cols,
    rows,
    branch_rule="phi1_above_eq",
    offset_index=1,
    eps_shift=1e-6,
    dt_sep=1e-3,
    steps_sep=1,
):
    total = len(params_x) #Общее число точек сетки
    dim = 4 #Размер фазового пространства

    inits = np.zeros(dim * total, dtype=np.float64) #Создаётся один длинный массив, куда будут записываться все начальные условия
    nones = [] #Сюда будут собираться индексы точек, где построить init не удалось
    eq_points = [None] * total #Здесь будут храниться равновесия, реально использованные для построения init
    branch_ids = [None] * total #Здесь хранятся номера выбранных ветвей (+1 или -1)
    unstable_dirs = [None] * total #Здесь будут храниться найденные неустойчивые направления

    #Перебираются все точки сетки
    for j in range(rows):
        for i in range(cols):
            idx = i + j * cols #Преобразование двумерного индекса (i, j) в одномерный индекс idx
            eq = eq_grid[idx] #Берётся равновесие для текущей точки сетки

            if eq is None: #Если в этой точке сетки равновесие не было найдено, то индекс добавляется в nones
                nones.append(idx)
                continue

            params = np.array(def_params, dtype=float) #Создаётся копия базового вектора параметров
            params[param_to_index[param_x_name]] = params_x[idx] #В эту копию подставляются значения параметров для текущей точки сетки
            params[param_to_index[param_y_name]] = params_y[idx]

            gamma = float(params[param_to_index["gamma"]]) #Из вектора параметров извлекаются нужные gamma, lam, k
            lam   = float(params[param_to_index["lambda"]])
            k     = float(params[param_to_index["k"]])

            # опорное направление от соседей
            ref_unstable_dir = None #Пока опорного направления нет
            candidate_neighbor_indices = [] #Список соседей, у которых можно взять опорный вектор
            if i - 1 >= 0: #Добавляется левый сосед, если он существует
                candidate_neighbor_indices.append((i - 1) + j * cols)
            if j - 1 >= 0: #Добавляется верхний сосед, если он существует
                candidate_neighbor_indices.append(i + (j - 1) * cols)

            #Идём по кандидатам-соседям и берём первое доступное опорное направление
            for nidx in candidate_neighbor_indices:
                if unstable_dirs[nidx] is not None:
                    ref_unstable_dir = unstable_dirs[nidx]
                    break

            try:
                result = build_init_from_equilibrium(
                    eq=eq, gamma=gamma, lam=lam, k=k, branch_rule=branch_rule,
                    offset_index=offset_index, eps_shift=eps_shift, dt_sep=dt_sep,
                    steps_sep=steps_sep, ref_unstable_dir=ref_unstable_dir, )

                inits[idx * dim:(idx + 1) * dim] = result["init_point"]
                eq_points[idx] = result["eq_point"]
                branch_ids[idx] = result["branch_id"]
                unstable_dirs[idx] = result["unstable_dir"]

            except Exception as e: #Если построить init не удалось, управление переходит сюда
                nones.append(idx) #Индекс текущей точки добавляется в список неудачных
                eq_points[idx] = None #Для этой точки зануляются сопутствующие результаты
                branch_ids[idx] = None
                unstable_dirs[idx] = None
                print(
                    f"[INIT FROM EQ GRID] idx={idx}, "
                    f"{param_x_name}={params_x[idx]:.6f}, "
                    f"{param_y_name}={params_y[idx]:.6f} -> failed: {e}"
                )

    return inits, np.array(nones, dtype=np.int32), eq_points


# превращает сетку параметров в сетку начальных условий на 1D unstable separatrix
def build_inits_on_parameter_grid_with_shape(
    params_x,
    params_y,
    def_params,
    param_x_name,
    param_y_name,
    param_to_index,
    cols,
    rows,
    center_i,
    center_j,
    saddle_focus_rule="phi1_lt_phi2",
    branch_rule="phi1_above_eq",
    offset_index=1,
    eps_shift=1e-4,
    dt_sep=1e-3,
    steps_sep=1,
):
    eq_grid = continue_target_equilibrium_on_grid( #строит eq_grid — сетку равновесий по всем точкам параметров
        params_x=params_x,
        params_y=params_y,
        def_params=def_params,
        param_x_name=param_x_name,
        param_y_name=param_y_name,
        param_to_index=param_to_index,
        cols=cols,
        rows=rows,
        center_i=center_i,
        center_j=center_j,
        saddle_focus_rule=saddle_focus_rule,
    )

    return build_inits_from_eq_grid( #строит init_point на сепаратрисе для каждой точки сетки (Большой массив длины 4 * total, где лежат все стартовые точки на сепаратрисах)
        eq_grid=eq_grid,
        params_x=params_x,
        params_y=params_y,
        def_params=def_params,
        param_x_name=param_x_name,
        param_y_name=param_y_name,
        param_to_index=param_to_index,
        cols=cols,
        rows=rows,
        branch_rule=branch_rule,
        offset_index=offset_index,
        eps_shift=eps_shift,
        dt_sep=dt_sep,
        steps_sep=steps_sep,
    )