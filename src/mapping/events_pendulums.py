import numpy as np

TWOPI = 2.0 * np.pi

def wrap_to_pi(a: float) -> float:
    """[-pi, pi)"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def crosses(a_prev: float, a_curr: float, level: float = 0.0, direction: int = 0) -> bool:
    """
    Проверка пересечения уровня `level`.
    direction:
      0  — любое пересечение
      +1 — снизу вверх
      -1 — сверху вниз
    """
    if direction == 0:
        return (a_prev - level) * (a_curr - level) < 0
    if direction > 0:
        return (a_prev < level) and (a_curr > level)
    return (a_prev > level) and (a_curr < level)

def detect_event_0_7(prev_state, curr_state) -> int:
    """
    prev_state, curr_state: [fi1, v1, fi2, v2]
    Возвращает:
      0..7  — если событие произошло
      -1    — если нет
    События :
      0: fi1 максимум  (v1: + -> -)
      1: fi1 минимум   (v1: - -> +)
      2: fi1 пересёк +pi вверх (: -pi -> +pi, т.е. -π/π разрыв)
      3: fi1 пересёк -pi вниз  (wrap: +pi -> -pi)

      4: fi2 максимум  (v2: + -> -)
      5: fi2 минимум   (v2: - -> +)
      6: fi2 пересёк +pi вверх (wrap: -pi -> +pi)
      7: fi2 пересёк -pi вниз  (wrap: +pi -> -pi)
    """

    fi1_prev, v1_prev, fi2_prev, v2_prev = prev_state
    fi1_curr, v1_curr, fi2_curr, v2_curr = curr_state

    # 0/1: экстремумы fi1 по смене знака v1
    if (v1_prev > 0.0) and (v1_curr < 0.0):
        return 0
    if (v1_prev < 0.0) and (v1_curr > 0.0):
        return 1

    # 4/5: экстремумы fi2 по смене знака v2
    if (v2_prev > 0.0) and (v2_curr < 0.0):
        return 4
    if (v2_prev < 0.0) and (v2_curr > 0.0):
        return 5

    # 2/3 и 6/7: пересечения "±pi" делаем через wrap_to_pi
    w1_prev = wrap_to_pi(fi1_prev)
    w1_curr = wrap_to_pi(fi1_curr)
    w2_prev = wrap_to_pi(fi2_prev)
    w2_curr = wrap_to_pi(fi2_curr)

    # Если wrap резко перепрыгнул через границу, это и есть событие.
    # -pi -> +pi  (рост)  : w_prev ~ -pi, w_curr ~ +pi
    # +pi -> -pi  (падение): w_prev ~ +pi, w_curr ~ -pi
    # В числах это выглядит как скачок примерно на 2*pi.
    if (w1_prev < -2.5) and (w1_curr >  2.5):
        return 2
    if (w1_prev >  2.5) and (w1_curr < -2.5):
        return 3
    if (w2_prev < -2.5) and (w2_curr >  2.5):
        return 6
    if (w2_prev >  2.5) and (w2_curr < -2.5):
        return 7

    return -1
