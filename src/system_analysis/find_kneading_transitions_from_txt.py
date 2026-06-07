import re
import os

# 1. ПУТИ К ФАЙЛАМ
txt_path = r"C:/Lobach4/tu/kneadings-master1/output/kneadings_pendulums1.txt"
# Файл, куда запишем результат (создастся в той же папке)
out_path = r"C:/Lobach4/tu/kneadings-master1/output/found_transitions.txt"

# Паттерн строго под диапазон событий от 00 до 24
pattern = re.compile(
    r"k:\s*([0-9.]+),\s*gamma:\s*([0-9.]+)\s*=>\s*([0-4]{2}(?:\s*-\s*[0-4]{2})*)\s*\(Raw:\s*([0-9.eE+-]+)\)"
)

points = []

# Читаем входной файл
if not os.path.exists(txt_path):
    print(f"ОШИБКА: Файл {txt_path} не найден!")
    exit()

print(f"Читаю файл: {txt_path}...")
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            k = float(m.group(1))
            gamma = float(m.group(2))
            code = m.group(3)
            raw = float(m.group(4))
            points.append((gamma, k, code, raw))

# Группируем по gamma (строкам)
rows = {}
for gamma, k, code, raw in points:
    rows.setdefault(gamma, []).append((k, code, raw))

print(f"Обработано точек: {len(points)}. Ищу переходы...")

# Записываем результат в файл
with open(out_path, "w", encoding="utf-8") as out_f:
    out_f.write("ОТЧЕТ О НАЙДЕННЫХ ПЕРЕХОДАХ НИДИНГ-КОДОВ\n")
    out_f.write("=" * 80 + "\n\n")

    for row_num, gamma in enumerate(sorted(rows.keys())):
        row = sorted(rows[gamma], key=lambda x: x[0])

        row_transitions = []  # Сюда соберем переходы только для этой строки

        for i in range(len(row) - 1):
            k1, code1, raw1 = row[i]
            k2, code2, raw2 = row[i + 1]

            if code1 != code2:
                # Формируем строку с данными
                info = (
                    f"row≈{row_num:3}, gamma={gamma:.10f} | "
                    f"i={i:3}->{i + 1:3} | k={k1:.8f}->{k2:.8f} | "
                    f"code {code1} -> {code2} | raw {raw1:.4f} -> {raw2:.4f}"
                )
                row_transitions.append(info)

        # Если в строке есть переходы, пишем их в файл
        if row_transitions:
            out_f.write(f"СТРОКА {row_num} (gamma={gamma:.10f}): найдено {len(row_transitions)} переходов\n")
            for t in row_transitions:
                out_f.write(f"  {t}\n")
            out_f.write("-" * 40 + "\n")

print(f"\nГОТОВО! Список переходов сохранен в файл:\n{out_path}")