#читает текстовый файл kneadings_pendulums1.txt
#ищет места, где при изменении горизонтального параметра меняется код нидинга
#в данном виде работает для k и gamma

import re

txt_path = r"C:/Lobach4/tu/kneadings-master1/output/kneadings_pendulums1.txt" #файл с результатами

pattern = re.compile(
    r"k:\s*([0-9.]+),\s*gamma:\s*([0-9.]+)\s*=>\s*([0-7]+)\s*\(Raw:\s*([0-9.eE+-]+)\)"
)

points = []

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            k = float(m.group(1))
            gamma = float(m.group(2))
            code = m.group(3)
            raw = float(m.group(4))
            points.append((gamma, k, code, raw))

# группируем по gamma
rows = {}
for gamma, k, code, raw in points:
    rows.setdefault(gamma, []).append((k, code, raw))

print("НАЙДЕННЫЕ ПЕРЕХОДЫ ПО ГОРИЗОНТАЛЬНЫМ СТРОКАМ:\n")

for row_num, gamma in enumerate(sorted(rows.keys())):
    row = sorted(rows[gamma], key=lambda x: x[0])

    for i in range(len(row) - 1):
        k1, code1, raw1 = row[i]
        k2, code2, raw2 = row[i + 1]

        if code1 != code2:
            print(
                f"row≈{row_num}, gamma={gamma:.15f}, "
                f"i={i}->{i+1}, "
                f"k={k1:.15f}->{k2:.15f}, "
                f"code {code1}->{code2}, "
                f"raw {raw1}->{raw2}"
            )