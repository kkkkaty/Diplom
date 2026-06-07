import sys
sys.path.append(r"C:\Lobach4\Diplom\kneadings-master")
import numpy as np
import os
import matplotlib.pyplot as plt
from get_inits import find_equilibria_pendulum, equilibrium_type



def analyze_and_report_transition(config, kneading_map_flat, output_dir, row_idx, trans_num):

    """
    1. Находит границу на карте.
    2. Запускает сепаратрисы 'до' и 'после'.
    3. Находит седловое равновесие, об которое 'ударилась' сепаратриса.
    4. Строит графики и пишет отчет.
    """

    from src.system_analysis.analyze_separatrix_transition_pendulums_v2 import analyze_separatrix_transition_v2

    # Запуск основного анализа
    # closeness=0.0 означает, что мы берем точки максимально близко к границе (соседние пиксели)
    result = analyze_separatrix_transition_v2(
        config=config,
        kneading_map_flat=kneading_map_flat,
        output_dir=output_dir,
        row_index=row_idx,
        transition_number=trans_num,
        closeness=0.0
    )

    print(f"\n{'=' * 60}")
    print(f" АНАЛИЗ ГРАНИЦЫ")
    print(f"{'=' * 60}")

    # Проверяем, нашли ли мы седло, отвечающее за расхождение
    div_info = result.divergence_info
    if div_info and div_info['between_eq']:
        eq_data = div_info['between_eq']
        print(f"Причина смены нидинг-кода: сепаратриса проходит вблизи седла.")
        print(f"Координаты равновесия: {eq_data['eq_base']}")
        print(f"Тип: nU={eq_data['nU']} (неустойчивые), nS={eq_data['nS']} (устойчивые)")

        if eq_data['is_symmetric']:
            print("Это СИММЕТРИЧНОЕ состояние равновесия (phi1=phi2 или v1=v2=0).")
        else:
            print("Это асимметричное состояние равновесия.")

        print(f"Минимальное расстояние (log10): {eq_data['score']:.4f}")
        print(f"Момент сближения: t = {eq_data['time']:.2f}")
    else:
        print("Предупреждение: Явное разделяющее равновесие не найдено. Проверьте время интегрирования.")

    # Сохраняем текстовое резюме
    with open(os.path.join(output_dir, "diploma_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Анализ перехода на строке {row_idx}\n")
        f.write(f"Параметры перехода: {result.left_scan_point.x:.6f} -> {result.right_scan_point.x:.6f}\n")
        if div_info and div_info['between_eq']:
            f.write(f"Организующее равновесие: {div_info['between_eq']['eq_base']}\n")
            f.write(
                f"Тип (nU, nS, nC): {div_info['between_eq']['nU']}, {div_info['between_eq']['nS']}, {div_info['between_eq']['nC']}\n")

    return result


if __name__ == "__main__":
    import yaml

    # 1. ЗАГРУЗКА КОНФИГУРАЦИИ
    config_path = r"C:\Lobach4\Diplom\kneadings-master\config\kneadings_pendulums.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Не найден файл конфигурации по пути {config_path}")
        exit()

    # 2. ЗАГРУЗКА КАРТЫ НИДИНГОВ
    map_path = r"C:\Lobach4\tu\kneadings-master1\output\kneadings_pendulums1.npy"

    try:
        kneading_map = np.load(map_path)
    except FileNotFoundError:
        print(f"ОШИБКА: Не найден файл карты .npy по пути {map_path}")
        exit()

    # 3. НАСТРОЙКА АНАЛИЗА
    output_directory = r"C:\Lobach4\tu\kneadings-master1\output\result"

    # ВЫБОР ТОЧКИ НА КАРТЕ
    row_to_scan = 842  # Номер строки на карте (горизонтальный срез)
    trans_to_analyze = 0 # Номер перехода на этой строке

    # 4. ЗАПУСК
    print(f"--- Запуск анализа перехода на строке {row_to_scan} ---")

    analyze_and_report_transition(
        config=config_data,
        kneading_map_flat=kneading_map,
        output_dir=output_directory,
        row_idx=row_to_scan,
        trans_num=trans_to_analyze
    )

    print(f"Результаты здесь: {output_directory}")