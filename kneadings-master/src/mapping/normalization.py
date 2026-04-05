def get_period_length(kneading):
    kneading_len = len(kneading)
    period_found = True

    for curr_period in range(1, kneading_len // 2 + 1):
        period_found = True

        for i in range(curr_period, kneading_len - curr_period, curr_period):
            for j in range(curr_period):
                if kneading[j] != kneading[i + j]:
                    period_found = False
                    break
            if not period_found:
                break

        if period_found:
            period_len = curr_period
            # break
            return period_len # таким образом получим наименьший по длине паттерн

    period_len = kneading_len # если не нашли паттерн, значит, возвращаем всю последовательность
    # в случае построения карты хаоса нужно предусмотреть сжатие последовательности
    return period_len


# найти длину периода -> найти наименьший по сумме период ->
# заменить все числа в последовательности этим периодом
# для периодичности -- возможность отождествления (?)

def normalize_kneading(kneading):
    kneading_len = len(kneading)
    period_len = get_period_length(kneading)

    if period_len < kneading_len:

        pattern_min_10 = int(kneading, base=2)
        for shift in range(period_len): # цикл по сдвигу
            pattern = kneading[0:period_len]
            # for i in range(period_len): # цикл прохода по периоду для подсчёта его суммы
                # (1010)_10 = 1 * (1/2)^3 + 0 * (1/2)^2 + 1 * (1/2)^1 + 0 * (1/2)^0
                # curr_period += 2 * kneading[i]
            pattern = pattern[shift:] + pattern[:shift]
            # print(f"{pattern} при сдвиге {shift}")
            pattern_10 = int(pattern, base=2)
            pattern_min_10 = min(pattern_10, pattern_min_10)

        pattern_min = bin(pattern_min_10)[2:]
        if len(pattern_min) < period_len: # приводим к виду 00...011...1
            pattern_min = '0' * (period_len - len(pattern_min)) + pattern_min
        # print(f"Наименьший паттерн {pattern_min} при десятичном значении {pattern_min_10}")

        occurence_num = kneading_len // period_len # количество вхождений паттерна в последовательность целиком
        kneading_norm = pattern_min * occurence_num + pattern_min[:(kneading_len - occurence_num * period_len)]
        return kneading_norm, pattern_min

    return kneading, kneading


if __name__ == "__main__":
    kneading = '010010010'
    kneading_10 = int(kneading, base=2)
    period_len = get_period_length(kneading)
    print(f"Нидинг {kneading} длины {len(kneading)} с паттерном периода {period_len}")
    kneading_norm, kneading_min = normalize_kneading(kneading)
    print(kneading_norm, kneading_min)


