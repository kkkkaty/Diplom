import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Настройка шрифтов
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# 1. загрузка данных из отчета
json_path = r"C:/Lobach4/tu/kneadings-master1/output/manual_transition_check/transition_report.json"

if not os.path.exists(json_path):
    print(f"Ошибка: файл {json_path} не найден")
    print("Сначала нужно запустить manual_transition_check.py для генерации отчета")
    exit()

print(f"Загрузка отчета о переходе: {json_path}...")
with open(json_path, "r", encoding="utf-8") as f:
    report = json.load(f)

# извлекаем оранжевую и зеленую траектории
before_probe = [p for p in report["probes"] if p["label"] == "before"][0]
after_probe = [p for p in report["probes"] if p["label"] == "after"][0]

before_traj = np.array(before_probe["trajectory"]) # Массив координат (N, 4)
before_time = np.array(before_probe["time"])       # Массив времени (N)
after_traj = np.array(after_probe["trajectory"])
after_time = np.array(after_probe["time"])

# Считываем параметры системы (gamma, lambda, k)
gamma_val, lam_val, k_val = before_probe["params"]
print(f"Параметры системы: gamma={gamma_val}, lambda={lam_val}, k={k_val}")


# 2. функции системы и отображение Пуанкаре

def system(t, y, gamma, lam, k):
    phi1, v1, phi2, v2 = y
    return [
        v1,
        gamma - lam * v1 - np.sin(phi1) + k * np.sin(phi2 - phi1),
        v2,
        gamma - lam * v2 - np.sin(phi2) + k * np.sin(phi1 - phi2)
    ]

# Функция-датчик. Она возвращает v1. В момент пересечения нуля датчик срабатывает.
def event_v1_zero(t, y, gamma, lam, k):
    return y[1]  # v1


event_v1_zero.terminal = True # Требуем остановить расчет при срабатывании датчика
event_v1_zero.direction = 1 # Срабатывать только при пересечении нуля снизу вверх (v1 из минуса в плюс)
# Чтобы составить сечение Пуанкаре, нам нужен четкий триггер. Пересечение нуля скорости первого маятника снизу вверх — это и есть плоскость нашего сечения

def angle_diff(a, b):
    #Вычисляет разность углов с учетом периодичности на цилиндре
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def distance_poincare_points(u1, u2):
    #Вычисляет евклидово расстояние между точками Пуанкаре с учетом периодичности углов
    d_phi1 = angle_diff(u1[0], u2[0])
    d_phi2 = angle_diff(u1[1], u2[1])
    d_v2 = u1[2] - u2[2]
    return np.sqrt(d_phi1 ** 2 + d_phi2 ** 2 + d_v2 ** 2)

#Отображение Пуанкаре берет 3 координаты на сечении u=[fi1,fi2,v2], делает из них 4D старт (добавив v1=0), интегрирует маятники вперед по времени и ловит их в момент следующего возвращения на сечение.
def poincare_map(u, gamma, lam, k, t_offset=0.04, t_max=100.0, rtol=1e-10, atol=1e-10):
    phi1, phi2, v2 = u
    y0 = np.array([phi1, 0.0, phi2, v2], dtype=float)

    # Шаг вперед на 0.04 секунды без датчика, чтобы уйти с сечения v1=0 и не самоблокироваться
    sol_init = solve_ivp(system, [0.0, t_offset], y0, args=(gamma, lam, k), rtol=rtol, atol=atol)
    y_start = sol_init.y[:, -1]

    # Интегрируем до следующего пересечения v1 = 0 снизу вверх
    sol = solve_ivp(system, [t_offset, t_max], y_start, args=(gamma, lam, k), events=event_v1_zero, rtol=rtol, atol=atol)

    if not sol.t_events[0].size:
        return None, None # не вернулись на сечение

    y_return = sol.y[:, -1]
    # Заворачиваем вернувшиеся углы в диапазон [-pi, pi]
    phi1_ret = np.mod(y_return[0] + np.pi, 2 * np.pi) - np.pi
    phi2_ret = np.mod(y_return[2] + np.pi, 2 * np.pi) - np.pi
    v2_ret = y_return[3]
    T = sol.t_events[0][0] + t_offset

    return np.array([phi1_ret, phi2_ret, v2_ret]), T


# Отображение Пуанкаре кратности n (для поиска периодов n)
def poincare_map_n(u, n, gamma, lam, k, t_offset=0.04, t_max=100.0, rtol=1e-10, atol=1e-10):
    curr_u = u.copy()
    total_T = 0.0
    for _ in range(n):
        curr_u, T = poincare_map(curr_u, gamma, lam, k, t_offset=t_offset, t_max=t_max, rtol=rtol, atol=atol)
        if curr_u is None:
            return None, None
        total_T += T
    return curr_u, total_T

# Функция невязки: возвращает вектор расстояния между P^n(u) и u
def poincare_residual_n(u, n, gamma, lam, k, t_offset=0.04, t_max=100.0, rtol=1e-10, atol=1e-10):
    # Запускаем маятники с финиша из точки u и катим их n кругов вперед по времени
    p_u, _ = poincare_map_n(u, n, gamma, lam, k, t_offset=t_offset, t_max=t_max, rtol=rtol, atol=atol)
    # Если маятники улетели в бесконечность или застряли и не вернулись на финиш
    if p_u is None:
        # Возвращаем огромный штраф [1000, 1000, 1000], чтобы решатель ушел из этой опасной зоны
        return np.array([1e3, 1e3, 1e3])
    # 3. Если вернулись, считаем разницу (невязку) между финишем P^n(u) и стартом u
    res = np.zeros(3)
    # Используем функцию angle_diff, чтобы разность углов считалась правильно на окружности
    res[0] = angle_diff(p_u[0], u[0]) # Разница угла fi1
    res[1] = angle_diff(p_u[1], u[1]) # Разница угла fi2
    res[2] = p_u[2] - u[2]            # Разница скорости v2
    return res
#Это математическое описание неподвижной точки P^n(u)-u=0. Решая его, мы находим точные координаты замкнутых орбит любого периода n


# 3. Якобиан отображения Пуанкаре - это матрица размером 3 на 3, которая показывает, насколько чуствительна наша финишная точка P^n(u)
# к микроскопическим изменениям стартовых координат u. Собственные значения этой матрицы - МУЛЬТИПЛИКАТОРЫ ФЛОКЕ, которые доказывают
# какой именно предельный цикл, устойчивый или седловой.
def compute_poincare_jacobian_n(u_star, n, gamma, lam, k, eps=1e-7):
    J = np.zeros((3, 3))
    # берем найденную точку цикла du и по очереди для каждой из трех ее координат (fi1, fi2, v2) делаем маленький сдвиг на eps=10^-7
    for j in range(3):
        du = np.zeros(3)
        du[j] = eps

        # смещаем стартовую точку чуть-чуть вправо и пускаем маятники на n кругов. Записываем финишную точку p_plus
        p_plus, _ = poincare_map_n(u_star + du, n, gamma, lam, k)
        # смещаем стартовую точку чуть-чуть влево и пускаем маятники на n кругов. Записываем финишную точку p_minus
        p_minus, _ = poincare_map_n(u_star - du, n, gamma, lam, k)

        #Если при сдвиге траектория улетела в бесконечность и не вернулась, возвращаем None
        if p_plus is None or p_minus is None:
            return None

        # Мы считаем разницу между результатами финиша справа и слева по всем трем координатам (для углов обязательно используем нашу функцию angle_diff)
        d_phi1 = angle_diff(p_plus[0], p_minus[0])
        d_phi2 = angle_diff(p_plus[1], p_minus[1])
        d_v2 = p_plus[2] - p_minus[2]

        #делим эту разность на 2*eps, мы получаем численное значение производной (скорости изменения) для j столбца матрицы Якоби J
        #dP^n(u)/du_j=(P^n(u+du)-P^n(u-du))/2*eps
        J[:, j] = np.array([d_phi1, d_phi2, d_v2]) / (2 * eps)
    return J


#  4. сбор пересечений с обеих траекторий
def get_crossings_from_traj(traj, t_arr):
    crossings = []
    for i in range(1, len(traj)):
        v1_prev = traj[i - 1][1]
        v1_curr = traj[i][1]
        if v1_prev < 0 and v1_curr > 0:
            frac = -v1_prev / (v1_curr - v1_prev)
            #Точка пересечения скорости v1=0  почти никогда не совпадает с шагами идеально — она лежит где-то между шагом i-1 и шагом i
            #Чтобы найти координаты на финише со стопроцентной точностью, мы рассчитываем пропорцию (дробь frac) — в каком именно месте между двумя шагами скорость обратилась в ноль.
            p1 = traj[i - 1][0] + frac * (traj[i][0] - traj[i - 1][0])
            p2 = traj[i - 1][2] + frac * (traj[i][2] - traj[i - 1][2])
            v2 = traj[i - 1][3] + frac * (traj[i][3] - traj[i - 1][3])
            t_c = t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1])

            p1 = np.mod(p1 + np.pi, 2 * np.pi) - np.pi
            p2 = np.mod(p2 + np.pi, 2 * np.pi) - np.pi
            crossings.append((t_c, np.array([p1, p2, v2])))
    return crossings


all_crossings = []
all_crossings.extend(get_crossings_from_traj(before_traj, before_time))
all_crossings.extend(get_crossings_from_traj(after_traj, after_time))
print(f"Всего найдено кандидатов на сечении: {len(all_crossings)}")

# 5. поиск периодических орбит кратности 1, 2, 3
saddle_cycle_u = None
saddle_cycle_period = 0.0
saddle_cycle_n = 0
saddle_cycle_eigvec = None

print("\nПоиск периодических орбит кратности n = 1, 2, 3...")

found_cycles = []

#Предельный цикл не обязан быть простым (периода 1). Он может быть сложным, совершая несколько витков (например, 2 или 3 круга)
# перед тем, как окончательно замкнуться
for n in [1, 2, 3]:
    print(f"  [Циклы периода {n}]...")
    #мы перебираем все пересечения all_crossings, которые собрали на прошлом шаге с оранжевой и зеленой траекторий
    for idx, (t_c, u_guess) in enumerate(all_crossings):
        #Мы пропускаем первые и последние 5 пересечений, так как на самом старте траектория еще слишком близка к седло-фокусу,
        # а на самом финише она уже улетела на аттрактор. Мы ищем пересечения в середине спирали,
        # где траектория максимально близка к нашему циклу
        if idx < 5 or idx > len(all_crossings) - 5: continue

        #мы передаем наше стартовое приближение u_guess в решатель. он начинает двигать координаты (fi1,fi2,v2), запуская
        #отображение Пуанкаре P^n(u), пока разница между финишем и стартом не станет равна нулю.
        sol = root(lambda u: poincare_residual_n(u, n, gamma_val, lam_val, k_val), u_guess, method='hybr', tol=1e-8)

        if sol.success:
            u_star = sol.x
            res_val = np.linalg.norm(poincare_residual_n(u_star, n, gamma_val, lam_val, k_val))
            #Если решатель сошелся (sol.success равен True) и итоговая математическая невязка микроскопически мала (res_val < 1e-5),
            # мы считаем, что потенциальная замкнутая орбита найдена
            if res_val < 1e-5:
                # Проверка на уникальность с учетом периодичности углов
                is_new = True
                for old_u, old_n in found_cycles:
                    #С помощью нашей функции distance_poincare_points мы измеряем расстояние до уже найденных ранее циклов.
                    # Если такое место на цилиндре мы уже находили, мы помечаем кандидата как дубликат и переходим к следующему шагу
                    if distance_poincare_points(u_star, old_u) < 1e-3 and n == old_n:
                        is_new = False
                        break
                #Если цикл действительно уникален, мы рассчитываем его точный период T_star и сохраняем его координаты и период
                # в наш список уникальных циклов found_cycles
                if is_new:
                    _, T_star = poincare_map_n(u_star, n, gamma_val, lam_val, k_val)
                    found_cycles.append((u_star, n))

                    # Для каждого уникального цикла мы рассчитываем матрицу Якоби J и находим её собственные значения (мультипликаторы Флоке)
                    J = compute_poincare_jacobian_n(u_star, n, gamma_val, lam_val, k_val)
                    if J is not None:
                        eigvals, eigvecs = np.linalg.eig(J)
                        unstable = [m for m in eigvals if abs(m) > 1.0001]
                        stable = [m for m in eigvals if abs(m) < 0.9999]
                        #Если у цикла ровно одно неустойчивое направление (len(unstable) == 1) и два устойчивых (len(stable) == 2),
                        # этот цикл объявляется седловым
                        is_saddle = len(unstable) == 1 and len(stable) == 2
                        print(
                            f"  Найдена орбита периода {n}: T = {T_star:.4f} c. Мультипликаторы: {np.abs(eigvals)}")
                        #Если седловой цикл найден впервые, мы сохраняем его координаты, период и неустойчивый собственный вектор
                        # saddle_cycle_eigvec (направление, в котором траектория разлетается от седла)
                        if is_saddle and saddle_cycle_u is None:
                            saddle_cycle_u = u_star
                            saddle_cycle_period = T_star
                            saddle_cycle_n = n
                            # Извлекаем неустойчивый собственный вектор
                            unstable_idx = np.argmax(np.abs(eigvals))
                            saddle_cycle_eigvec = np.real(eigvecs[:, unstable_idx])

# 6. Верификация седлового цикла (протягивание возмущений)
if saddle_cycle_u is not None:
    print(f"\nНайден седловой цикл периода-{saddle_cycle_n}!")
    print(
        f"  Координаты на сечении: phi1={saddle_cycle_u[0]:.6f}, phi2={saddle_cycle_u[1]:.6f}, v2={saddle_cycle_u[2]:.6f}")
    print(f"  Период: T = {saddle_cycle_period:.4f} секунд")

    # Нормируем неустойчивый собственный вектор (делаем его длину равной 1)
    v_u = saddle_cycle_eigvec / np.linalg.norm(saddle_cycle_eigvec)

    # Делаем микроскопический сдвиг на величину 0.0001 вдоль этого вектора в обе стороны
    delta_shift = 1e-4
    u_plus = saddle_cycle_u + delta_shift * v_u
    u_minus = saddle_cycle_u - delta_shift * v_u

    # Собираем полноценные 4D стартовые точки (занулив скорость v1)
    y0_plus = np.array([u_plus[0], 0.0, u_plus[1], u_plus[2]])
    y0_minus = np.array([u_minus[0], 0.0, u_minus[1], u_minus[2]])
    #Мы получили две стартовые точки y0_plus и y0_minus, лежащие по разные стороны от нашего седлового цикла

    print("\n Интегрирование двух ветвей неустойчивого многообразия седлового цикла...")
    #Интегрируем обе ветви вперед по времени на 150 секунд
    sol_plus = solve_ivp(system, [0, 150], y0_plus, args=(gamma_val, lam_val, k_val), rtol=1e-10, atol=1e-10,
                         max_step=0.05)
    sol_minus = solve_ivp(system, [0, 150], y0_minus, args=(gamma_val, lam_val, k_val), rtol=1e-10, atol=1e-10,
                          max_step=0.05)

    # Интегрируем сам седловой цикл ровно за один период, чтобы нарисовать его замкнутой синей петлей
    y0_cycle = np.array([saddle_cycle_u[0], 0.0, saddle_cycle_u[1], saddle_cycle_u[2]])
    sol_cycle = solve_ivp(system, [0, saddle_cycle_period], y0_cycle, args=(gamma_val, lam_val, k_val), rtol=1e-11,
                          atol=1e-11, max_step=0.01)
    #Мы проинтегрировали две ветви неустойчивого многообразия W+^u и W-^u

    # 7. График
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    proj = [((0, 2), r"$(\phi_1, \phi_2)$", r"$\phi_1$", r"$\phi_2$"),
            ((1, 3), r"$(v_1, v_2)$", r"$v_1$", r"$v_2$"),
            ((0, 1), r"$(\phi_1, v_1)$", r"$\phi_1$", r"$v_1$"),
            ((2, 3), r"$(\phi_2, v_2)$", r"$\phi_2$", r"$v_2$")]

    # Палитра
    c_plus = '#FF8000'  # Оранжевый (Ветвь +)
    c_minus = '#00CC44'  # Зеленый (Ветвь -)
    c_cycle = '#0000FF'  # Синий (Седловой цикл)

    for idx, (ax, ((ii, jj), title, xl, yl)) in enumerate(zip(axes.ravel(), proj)):
        # Отрисовка ветвей неустойчивого многообразия седлового цикла
        ax.plot(sol_plus.y[ii], sol_plus.y[jj], color=c_plus, lw=1.2, label="Ветвь многообразия $W^u_+$")
        ax.plot(sol_minus.y[ii], sol_minus.y[jj], color=c_minus, lw=1.2, label="Ветвь многообразия $W^u_-$")

        # Сам разделяющий седловой цикл
        ax.plot(sol_cycle.y[ii], sol_cycle.y[jj], color=c_cycle, lw=2.5, label="Седловой цикл", zorder=10)
        ax.scatter(y0_cycle[ii], y0_cycle[jj], color='red', marker='*', s=150, zorder=11, label="Точка Пуанкаре")

        ax.set_title(title, fontsize=11, pad=6, fontweight='bold')
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.2)

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

        if (ii, jj) == (0, 2):
            ax.set_xlim(1.79, 1.83);
            ax.set_ylim(0, 100)
        elif (ii, jj) == (1, 3):
            ax.set_xlim(-0.05, 0.1);
            ax.set_ylim(4, 6)
        elif (ii, jj) == (0, 1):
            ax.set_xlim(1.2, 2.2);
            ax.set_ylim(-0.11, 0.15)
        elif (ii, jj) == (2, 3):
            ax.set_xlim(0, 100);
            ax.set_ylim(4, 6)

    plt.suptitle(f"$k={k_val}$, $\\gamma={gamma_val}$, $\\lambda={lam_val}$", fontsize=13, y=0.95, fontweight='bold')

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.02), frameon=True,
               edgecolor='lightgrey')
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.86, wspace=0.24, hspace=0.32)

    out_img_path = os.path.join(report["output_dir"], "07_separating_limit_cycle_verified.png")
    plt.savefig(out_img_path, dpi=300)
    print(f"\n График сохранен в:\n{out_img_path}")
    plt.show()

else:
    print("\nСедловой предельный цикл среди кандидатов не обнаружен.")