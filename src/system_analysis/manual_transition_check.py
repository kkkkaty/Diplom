#выбираем какой переход между цветами анализировать вручную
#row_index - какую строку карты берем
#left_i, right_i - соседние точки, где мы увидели смену цвета


import os
import yaml
import numpy as np


from src.system_analysis.analyze_separatrix_transition_pendulums_v2 import (
    analyze_separatrix_transition_v2,
    extract_horizontal_scan,
    find_code_transitions_on_scan,
)

config_path = r"/config/kneadings_pendulums.yaml"
npy_path = r"C:/Lobach4/tu/kneadings-master1/output/kneadings_pendulums1.npy"

output_dir = r"C:/Lobach4/tu/kneadings-master1/output/manual_transition_check"

# сюда вручную вписываем строку карты
row_index = 416

# сюда вручную вписываем две соседние точки, между которыми сменился цвет
left_i = 662
right_i = 663

# сколько считать траекторию
n_steps_traj = 30000
stride_traj = 1


with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

kneading_map = np.load(npy_path)

scan = extract_horizontal_scan(
    kneading_map_flat=kneading_map,
    config=config,
    row_index=row_index,
)

transitions = find_code_transitions_on_scan(scan)

target_pair = (left_i, right_i)

if target_pair not in transitions:
    print("Такой переход не найден.")
    print("Проверь row_index, left_i, right_i.")
    print("Ближайшие найденные переходы в этой строке:")
    print(transitions[:30])
    raise SystemExit

transition_number = transitions.index(target_pair)

print("Найден переход:")
print("row_index =", row_index)
print("transition_number =", transition_number)
print("left_i, right_i =", target_pair)
print("left x =", scan[left_i].x, "code =", scan[left_i].code)
print("right x =", scan[right_i].x, "code =", scan[right_i].code)

result = analyze_separatrix_transition_v2(
    config=config,
    kneading_map_flat=kneading_map,
    output_dir=output_dir,
    row_index=row_index,
    transition_number=transition_number,
    closeness=0.01,
    dt_traj=None,
    n_steps_traj=n_steps_traj,
    stride_traj=stride_traj,
    prefer_symmetric_equilibria=False,
)

scan = extract_horizontal_scan(
    kneading_map_flat=kneading_map,
    config=config,
    row_index=200,   # любую строку
)

transitions = find_code_transitions_on_scan(scan)

print("ВСЕ переходы в этой строке:\n")

for idx, (l, r) in enumerate(transitions):
    print(
        idx,
        "| i =", l, "->", r,
        "| x =", scan[l].x, "->", scan[r].x,
        "| code:", scan[l].code, "->", scan[r].code,
    )
print("\nГотово.")
print("Папка с результатами:")
print(result.output_dir)