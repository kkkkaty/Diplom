import math
import numpy as np

from src.mapping.events_pendulums import detect_event_0_7


# -----------------------------
# ПАРАМЕТРЫ (можешь менять)
# -----------------------------
gamma = 0.8
lam = 0.1
k = 0.2

dt = 0.01
steps = 20000

# начальное состояние (можешь менять)
y0 = np.array([2.8, -3.0, 2.9, -2.5], dtype=np.float64)

EPS = 1e-12


# -----------------------------
# СИСТЕМА (как в CUDA)
# -----------------------------
def rhs(y):
    fi1, v1, fi2, v2 = y

    dfi1 = v1
    dv1 = -lam * v1 - math.sin(fi1) + gamma + k * math.sin(fi2 - fi1)

    dfi2 = v2
    dv2 = -lam * v2 - math.sin(fi2) + gamma + k * math.sin(fi1 - fi2)

    return np.array([dfi1, dv1, dfi2, dv2], dtype=np.float64)


# -----------------------------
# RK4 интегратор
# -----------------------------
def rk4_step(y, dt):
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)

    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# СЧИТАЕМ ТРАЕКТОРИЮ
# -----------------------------
def compute_trajectory(y0, steps, dt):
    traj = np.zeros((steps, 4), dtype=np.float64)
    traj[0] = y0

    for i in range(1, steps):
        traj[i] = rk4_step(traj[i - 1], dt)

    return traj


# -----------------------------
# CUDA-like детектор
# -----------------------------
def detect_event_cuda_like(prev, curr):
    fi1_prev, v1_prev, fi2_prev, v2_prev = prev
    fi1_curr, v1_curr, fi2_curr, v2_curr = curr

    c1_prev = math.cos(0.5 * fi1_prev)
    c1_curr = math.cos(0.5 * fi1_curr)
    c2_prev = math.cos(0.5 * fi2_prev)
    c2_curr = math.cos(0.5 * fi2_curr)

    s1_prev = -math.sin(fi1_prev)
    s1_curr = -math.sin(fi1_curr)
    s2_prev = -math.sin(fi2_prev)
    s2_curr = -math.sin(fi2_curr)

    def crossed_neg_to_pos(a_prev, a_curr):
        return (a_prev < -EPS) and (a_curr > EPS)

    def crossed_pos_to_neg(a_prev, a_curr):
        return (a_prev > EPS) and (a_curr < -EPS)

    cross1 = crossed_neg_to_pos(c1_prev, c1_curr) or crossed_pos_to_neg(c1_prev, c1_curr)
    cross2 = crossed_neg_to_pos(c2_prev, c2_curr) or crossed_pos_to_neg(c2_prev, c2_curr)

    # fi1
    if (v1_prev > EPS) and (v1_curr < -EPS):
        return 0
    if (v1_prev < -EPS) and (v1_curr > EPS):
        return 1
    if cross1 and (s1_prev < -EPS) and (s1_curr > EPS):
        return 2
    if cross1 and (s1_prev > EPS) and (s1_curr < -EPS):
        return 3

    # fi2
    if (v2_prev > EPS) and (v2_curr < -EPS):
        return 4
    if (v2_prev < -EPS) and (v2_curr > EPS):
        return 5
    if cross2 and (s2_prev < -EPS) and (s2_curr > EPS):
        return 6
    if cross2 and (s2_prev > EPS) and (s2_curr < -EPS):
        return 7

    return -1


# -----------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -----------------------------
def extract_events(func, traj):
    events = []
    for i in range(1, len(traj)):
        evt = func(traj[i - 1], traj[i])
        if evt != -1:
            events.append(evt)
    return events


# -----------------------------
# ЗАПУСК ТЕСТА
# -----------------------------
if __name__ == "__main__":
    print("Computing trajectory...")
    traj = compute_trajectory(y0, steps, dt)

    print("Extracting CPU events...")
    cpu_events = extract_events(detect_event_0_7, traj)

    print("Extracting CUDA-like events...")
    cuda_events = extract_events(detect_event_cuda_like, traj)

    print("\n--- RESULTS ---")
    print("CPU events count:", len(cpu_events))
    print("CUDA-like events count:", len(cuda_events))

    equal = (cpu_events == cuda_events)
    print("Sequences equal:", equal)

    if not equal:
        print("\nFirst mismatch:")

        for i, (e1, e2) in enumerate(zip(cpu_events, cuda_events)):
            if e1 != e2:
                print(f"Index {i}: CPU={e1}, CUDA={e2}")
                break

    print("\nFirst 50 events:")
    print("CPU      :", cpu_events[:50])
    print("CUDA-like:", cuda_events[:50])

    import math
    import numpy as np

    from src.mapping.events_pendulums import detect_event_0_7

    EPS = 1e-12


    def detect_event_cuda_like(prev, curr):
        fi1_prev, v1_prev, fi2_prev, v2_prev = prev
        fi1_curr, v1_curr, fi2_curr, v2_curr = curr

        c1_prev = math.cos(0.5 * fi1_prev)
        c1_curr = math.cos(0.5 * fi1_curr)
        c2_prev = math.cos(0.5 * fi2_prev)
        c2_curr = math.cos(0.5 * fi2_curr)

        s1_prev = -math.sin(fi1_prev)
        s1_curr = -math.sin(fi1_curr)
        s2_prev = -math.sin(fi2_prev)
        s2_curr = -math.sin(fi2_curr)

        def crossed_neg_to_pos(a_prev, a_curr):
            return (a_prev < -EPS) and (a_curr > EPS)

        def crossed_pos_to_neg(a_prev, a_curr):
            return (a_prev > EPS) and (a_curr < -EPS)

        cross1 = crossed_neg_to_pos(c1_prev, c1_curr) or crossed_pos_to_neg(c1_prev, c1_curr)
        cross2 = crossed_neg_to_pos(c2_prev, c2_curr) or crossed_pos_to_neg(c2_prev, c2_curr)

        if (v1_prev > EPS) and (v1_curr < -EPS):
            return 0
        if (v1_prev < -EPS) and (v1_curr > EPS):
            return 1
        if cross1 and (s1_prev < -EPS) and (s1_curr > EPS):
            return 2
        if cross1 and (s1_prev > EPS) and (s1_curr < -EPS):
            return 3

        if (v2_prev > EPS) and (v2_curr < -EPS):
            return 4
        if (v2_prev < -EPS) and (v2_curr > EPS):
            return 5
        if cross2 and (s2_prev < -EPS) and (s2_curr > EPS):
            return 6
        if cross2 and (s2_prev > EPS) and (s2_curr < -EPS):
            return 7

        return -1


    def print_single_check(name, prev_state, curr_state):
        cpu_evt = detect_event_0_7(prev_state, curr_state)
        cuda_evt = detect_event_cuda_like(prev_state, curr_state)

        print(f"\n{name}")
        print("prev:", prev_state)
        print("curr:", curr_state)
        print("CPU event      :", cpu_evt)
        print("CUDA-like event:", cuda_evt)
        print("Equal          :", cpu_evt == cuda_evt)

        fi1_prev, _, fi2_prev, _ = prev_state
        fi1_curr, _, fi2_curr, _ = curr_state

        print("c1 prev/curr:", math.cos(0.5 * fi1_prev), math.cos(0.5 * fi1_curr))
        print("s1 prev/curr:", -math.sin(fi1_prev), -math.sin(fi1_curr))
        print("c2 prev/curr:", math.cos(0.5 * fi2_prev), math.cos(0.5 * fi2_curr))
        print("s2 prev/curr:", -math.sin(fi2_prev), -math.sin(fi2_curr))


    if __name__ == "__main__":
        # ---------- Явный тест на событие 2 ----------
        # fi1: около pi, -sin: - -> +, cos(fi1/2) меняет знак
        prev_2 = np.array([math.pi - 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        curr_2 = np.array([math.pi + 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        print_single_check("Expected event 2", prev_2, curr_2)

        # ---------- Явный тест на событие 3 ----------
        # fi1: обратный проход через pi
        prev_3 = np.array([math.pi + 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        curr_3 = np.array([math.pi - 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        print_single_check("Expected event 3", prev_3, curr_3)

        # ---------- Явный тест на событие 6 ----------
        prev_6 = np.array([0.0, 0.0, math.pi - 0.1, 0.0], dtype=np.float64)
        curr_6 = np.array([0.0, 0.0, math.pi + 0.1, 0.0], dtype=np.float64)
        print_single_check("Expected event 6", prev_6, curr_6)

        # ---------- Явный тест на событие 7 ----------
        prev_7 = np.array([0.0, 0.0, math.pi + 0.1, 0.0], dtype=np.float64)
        curr_7 = np.array([0.0, 0.0, math.pi - 0.1, 0.0], dtype=np.float64)
        print_single_check("Expected event 7", prev_7, curr_7)