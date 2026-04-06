import numpy as np

EPS = 1e-12

#функции перехода через ноль
def crossed_neg_to_pos(a_prev: float, a_curr: float) -> bool:
    return (a_prev < -EPS) and (a_curr > EPS)
def crossed_pos_to_neg(a_prev: float, a_curr: float) -> bool:
    return (a_prev > EPS) and (a_curr < -EPS)


def detect_event_0_7(prev_state, curr_state) -> int:
    """
    Возвращает:
        0..7 — если событие произошло
        -1   — если нет
    События:
        0: max fi1   : v1 + -> -
        1: min fi1   : v1 - -> +
        2: jump up   : -sin(fi1) - -> + и cos(fi1/2) меняет знак
        3: jump down : -sin(fi1) + -> - и cos(fi1/2) меняет знак
        4: max fi2   : v2 + -> -
        5: min fi2   : v2 - -> +
        6: jump up   : -sin(fi2) - -> + и cos(fi2/2) меняет знак
        7: jump down : -sin(fi2) + -> - и cos(fi2/2) меняет знак
    """

    fi1_prev, v1_prev, fi2_prev, v2_prev = prev_state
    fi1_curr, v1_curr, fi2_curr, v2_curr = curr_state

    c1_prev = np.cos(0.5 * fi1_prev)
    c1_curr = np.cos(0.5 * fi1_curr)
    c2_prev = np.cos(0.5 * fi2_prev)
    c2_curr = np.cos(0.5 * fi2_curr)

    s1_prev = -np.sin(fi1_prev)
    s1_curr = -np.sin(fi1_curr)
    s2_prev = -np.sin(fi2_prev)
    s2_curr = -np.sin(fi2_curr)

    cross1 = crossed_neg_to_pos(c1_prev, c1_curr) or crossed_pos_to_neg(c1_prev, c1_curr)
    cross2 = crossed_neg_to_pos(c2_prev, c2_curr) or crossed_pos_to_neg(c2_prev, c2_curr)

    # 0/1: экстремумы fi1
    if (v1_prev > EPS) and (v1_curr < -EPS):
        return 0

    if (v1_prev < -EPS) and (v1_curr > EPS):
        return 1

    # 2/3: jump-события fi1
    if cross1 and (s1_prev < -EPS) and (s1_curr > EPS):
        return 2

    if cross1 and (s1_prev > EPS) and (s1_curr < -EPS):
        return 3

    # 4/5: экстремумы fi2
    if (v2_prev > EPS) and (v2_curr < -EPS):
        return 4

    if (v2_prev < -EPS) and (v2_curr > EPS):
        return 5

    # 6/7: jump-события fi2
    if cross2 and (s2_prev < -EPS) and (s2_curr > EPS):
        return 6

    if cross2 and (s2_prev > EPS) and (s2_curr < -EPS):
        return 7

    return -1