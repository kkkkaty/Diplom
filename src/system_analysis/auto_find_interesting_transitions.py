# src/system_analysis/auto_find_interesting_transitions.py
# -*- coding: utf-8 -*-

import os
import json
import yaml
import numpy as np

from src.system_analysis.analyze_separatrix_transition_pendulums_v2 import (
    analyze_separatrix_transition_v2,
    extract_horizontal_scan,
    find_code_transitions_on_scan,
)


# =========================================================
# ПУТИ
# =========================================================

config_path = r"C:\Lobach4\Diplom\kneadings-master\config\kneadings_pendulums.yaml"
npy_path = r"C:\Lobach4\tu\kneadings-master1\output\kneadings_pendulums1.npy"

output_root = r"C:\Lobach4\tu\kneadings-master1\output\auto_transition_search"
os.makedirs(output_root, exist_ok=True)


# =========================================================
# НАСТРОЙКИ ПОИСКА
# =========================================================

# какие строки карты смотреть
row_start = 80
row_end = 150

# сколько переходов максимум проверять в каждой строке
max_transitions_per_row = None  # можно поставить 10, если долго

# время интегрирования
n_steps_traj = 100000
stride_traj = 1

# насколько близко брать точки к границе
closeness = 0.01

# главный критерий близости к равновесию:
# log10(distance) < -1 значит distance < 0.1
# log10(distance) < -2 значит distance < 0.01
distance_log_threshold = -1.0

# не хотим, чтобы минимум был только в самом конце траектории
# если минимум достигается на последнем шаге, значит траектория просто ещё не дошла
max_argmin_time_fraction = 0.85


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def load_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def probe_is_good(probe, total_time):
    """
    Проверяем одну траекторию: before или after.
    """
    dlog = probe.get("best_eq_min_log10", None)
    tmin = probe.get("best_eq_argmin_t", None)
    eq = probe.get("closest_eq", None)

    if dlog is None or tmin is None or eq is None:
        return False

    # траектория должна реально близко пройти к равновесию
    if dlog > distance_log_threshold:
        return False

    # минимум расстояния не должен быть в самом конце
    if tmin > max_argmin_time_fraction * total_time:
        return False

    return True


def analyze_report_quality(report):
    """
    Возвращает оценку интересности перехода.
    """
    probes = report.get("probes", [])
    if len(probes) < 2:
        return None

    before = probes[0]
    after = probes[1]

    total_time = n_steps_traj * report_dt(report)

    before_good = probe_is_good(before, total_time)
    after_good = probe_is_good(after, total_time)

    if not (before_good or after_good):
        return None

    before_eq = before.get("closest_eq")
    after_eq = after.get("closest_eq")

    before_d = before.get("best_eq_min_log10")
    after_d = after.get("best_eq_min_log10")

    # Чем меньше log10 расстояния, тем лучше
    best_d = min(
        before_d if before_d is not None else 999,
        after_d if after_d is not None else 999,
    )

    same_eq = False
    if before_eq is not None and after_eq is not None:
        same_eq = np.linalg.norm(np.array(before_eq) - np.array(after_eq)) < 1e-4

    return {
        "best_log_distance": best_d,
        "before_good": before_good,
        "after_good": after_good,
        "same_closest_eq": same_eq,
        "before_eq": before_eq,
        "after_eq": after_eq,
        "before_d": before_d,
        "after_d": after_d,
        "before_tmin": before.get("best_eq_argmin_t"),
        "after_tmin": after.get("best_eq_argmin_t"),
    }


def report_dt(report):
    """
    Если dt явно не записан в report, берём из параметров расчёта.
    В твоём конфиге dt = 0.01.
    """
    return 0.01


# =========================================================
# ОСНОВНОЙ ПОИСК
# =========================================================

def main():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kneading_map = np.load(npy_path)

    good_results = []

    for row_index in range(row_start, row_end + 1):
        print("\n" + "=" * 70)
        print(f"ROW {row_index}")

        scan = extract_horizontal_scan(
            kneading_map_flat=kneading_map,
            config=config,
            row_index=row_index,
        )

        transitions = find_code_transitions_on_scan(scan)

        if max_transitions_per_row is not None:
            transitions = transitions[:max_transitions_per_row]

        print(f"Найдено переходов в строке: {len(transitions)}")

        for transition_number, (left_i, right_i) in enumerate(transitions):
            left_code = scan[left_i].code
            right_code = scan[right_i].code

            print(
                f"Проверяю row={row_index}, "
                f"i={left_i}->{right_i}, "
                f"code {left_code}->{right_code}"
            )

            local_output_dir = os.path.join(
                output_root,
                f"row_{row_index:04d}_i_{left_i:04d}_{right_i:04d}"
            )

            try:
                result = analyze_separatrix_transition_v2(
                    config=config,
                    kneading_map_flat=kneading_map,
                    output_dir=local_output_dir,
                    row_index=row_index,
                    transition_number=transition_number,
                    closeness=closeness,
                    dt_traj=None,
                    n_steps_traj=n_steps_traj,
                    stride_traj=stride_traj,
                    prefer_symmetric_equilibria=False,
                )

                report_path = os.path.join(result.output_dir, "transition_report.json")

                if not os.path.exists(report_path):
                    print("  report не найден")
                    continue

                report = load_report(report_path)
                quality = analyze_report_quality(report)

                if quality is None:
                    print("  неинтересно")
                    continue

                item = {
                    "row_index": row_index,
                    "left_i": left_i,
                    "right_i": right_i,
                    "left_k": scan[left_i].x,
                    "right_k": scan[right_i].x,
                    "gamma": scan[left_i].y,
                    "left_code": left_code,
                    "right_code": right_code,
                    "output_dir": result.output_dir,
                    **quality,
                }

                good_results.append(item)

                print("  >>> ИНТЕРЕСНЫЙ ПЕРЕХОД!")
                print("      best_log_distance =", quality["best_log_distance"])
                print("      before_d =", quality["before_d"])
                print("      after_d  =", quality["after_d"])
                print("      output =", result.output_dir)

            except Exception as e:
                print("  ошибка:", repr(e))
                continue

    # сортируем: сначала самые близкие подходы к равновесию
    good_results.sort(key=lambda x: x["best_log_distance"])

    out_json = os.path.join(output_root, "interesting_transitions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(good_results, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(output_root, "interesting_transitions.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for idx, item in enumerate(good_results):
            f.write(f"#{idx}\n")
            f.write(f"row_index = {item['row_index']}\n")
            f.write(f"left_i = {item['left_i']}\n")
            f.write(f"right_i = {item['right_i']}\n")
            f.write(f"gamma = {item['gamma']}\n")
            f.write(f"k: {item['left_k']} -> {item['right_k']}\n")
            f.write(f"code: {item['left_code']} -> {item['right_code']}\n")
            f.write(f"best_log_distance = {item['best_log_distance']}\n")
            f.write(f"before_d = {item['before_d']}\n")
            f.write(f"after_d = {item['after_d']}\n")
            f.write(f"same_closest_eq = {item['same_closest_eq']}\n")
            f.write(f"output_dir = {item['output_dir']}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("Найдено интересных переходов:", len(good_results))
    print("JSON:", out_json)
    print("TXT:", out_txt)


if __name__ == "__main__":
    main()