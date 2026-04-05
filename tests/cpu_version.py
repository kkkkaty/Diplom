import numpy as np
from src.mapping.convert import decimal_to_number_system

DIM = 4
DIM_REDUCED = DIM - 1
THREADS_PER_BLOCK = 512

INFINITY = 10

KneadingDoNotEndError = -0.1
InfinityError = -0.2
NoInitFound = -0.3

def det4x4(m):
    det = 0.0
    sign = 1.0

    minor = [0.] * 9

    for col in range(4):
        minor_row_idx = 0
        for i in range(1, 4):
            minor_col_idx = 0
            for j in range(4):
                if j != col:
                    minor[minor_row_idx * 3 + minor_col_idx] = m[i * 4 + j]
                    minor_col_idx += 1
            minor_row_idx += 1

        det_minor = (
            minor[0] * (minor[4] * minor[8] - minor[5] * minor[7]) -
            minor[1] * (minor[3] * minor[8] - minor[5] * minor[6]) +
            minor[2] * (minor[3] * minor[7] - minor[4] * minor[6])
        )

        det += sign * m[0 * 4 + col] * det_minor
        sign *= -1.0
    return det


def bary_expansion(pt):
    """
    globalPtCoords must be a 3d vector with 0 <= x <= y <= z <= 2pi, i.e. inside a CIR
    returns an expansion of (globalPtCoords - center of mass) in barycentric coordinates
    """

    pt_o = [0.] * 3
    pt_a = [0.] * 3
    pt_b = [0.] * 3
    pt_c = [0.] * 3
    pt_w = [0.] * 3
    vec_wa = [0.] * 3
    vec_wb = [0.] * 3
    vec_wc = [0.] * 3
    vec_wo = [0.] * 3
    mat_bary = [0.] * 16
    rhs = [0.] * 4

    pt_o[0] = 0.0; pt_o[1] = 0.0; pt_o[2] = 0.0
    pt_a[0] = 0.0; pt_a[1] = 0.0; pt_a[2] = 2 * np.pi
    pt_b[0] = 0.0; pt_b[1] = 2 * np.pi; pt_b[2] = 2 * np.pi
    pt_c[0] = 2 * np.pi; pt_c[1] = 2 * np.pi; pt_c[2] = 2 * np.pi

    pt_w[0] = 0.25 * (pt_a[0] + pt_b[0] + pt_c[0] - 3 * pt_o[0])
    pt_w[1] = 0.25 * (pt_a[1] + pt_b[1] + pt_c[1] - 3 * pt_o[1])
    pt_w[2] = 0.25 * (pt_a[2] + pt_b[2] + pt_c[2] - 3 * pt_o[2])

    for i in range(3):
        vec_wa[i] = pt_a[i] - pt_w[i]
        vec_wb[i] = pt_b[i] - pt_w[i]
        vec_wc[i] = pt_c[i] - pt_w[i]
        vec_wo[i] = pt_o[i] - pt_w[i]

        mat_bary[4 * i] = vec_wa[i]
        mat_bary[4 * i + 1] = vec_wb[i]
        mat_bary[4 * i + 2] = vec_wc[i]
        mat_bary[4 * i + 3] = vec_wo[i]

        rhs[i] = pt[i] - pt_w[i]

    mat_bary[12] = 1.; mat_bary[13] = 1.; mat_bary[14] = 1.; mat_bary[15] = 1.
    rhs[3] = 1.

    main_det = det4x4(mat_bary)

    bary_coords = [0.] * 4

    if abs(main_det) < 1e-12:
        bary_coords[:] = 0.
        return bary_coords

    # заполняем координаты решая систему методом Крамера
    for col in range(4):
        modified_mat = mat_bary.copy()
        for row in range(4):
            modified_mat[4 * row + col] = rhs[row]

        coord_det = det4x4(modified_mat)
        bary_coords[col] = coord_det / main_det

    return bary_coords


def get_domain_num(bary_expansion):
    min_coord = bary_expansion[0]
    i = 0
    domain_num = i
    while i < 4:
        if bary_expansion[i] < min_coord:
            min_coord = bary_expansion[i]
            domain_num = i
        i += 1
    return domain_num


def full_rhs(params, phis):
    """Calculates the right-hand side of the full system"""
    w, a, b, r = params
    rhs_phis = [w] * 4
    for i in range(4):
        for j in range(4):
            rhs_phis[i] += 0.25 * (-np.sin(phis[i] - phis[j] + a) + r * np.sin(2 * (phis[i] - phis[j]) + b))
    return rhs_phis


def reduced_rhs(params, psis):
    """Calculates the right-hand side of the reduced system"""
    phis = [0.] + psis
    rhs_phis = full_rhs(params, phis)
    rhs_psis = [0.] * 4
    for i in range(4):
        rhs_psis[i] = rhs_phis[i] - rhs_phis[0]
    return rhs_psis[1:]


def avg_face_dist_deriv(params, pt):
    """Average distance from the point to the faces of the thetrahedron"""
    x, y, z = pt
    sys_curr = reduced_rhs(params, pt)  # добавил params
    afdd = (1.0*x - 0.5*y) * sys_curr[0] + (-0.5*x + 1.0*y - 0.5*z) * sys_curr[1] + (-0.5*y + 1.0*z - np.pi) * sys_curr[2]
    return afdd


def stepper_rk4(params, y_curr, dt):
    """Makes RK-4 step and saves the value in y_curr"""
    k1 = reduced_rhs(params, y_curr)

    y_temp = [y_curr[i] + k1[i] * dt / 2.0 for i in range(DIM_REDUCED)]
    k2 = reduced_rhs(params, y_temp)

    y_temp = [y_curr[i] + k2[i] * dt / 2.0 for i in range(DIM_REDUCED)]
    k3 = reduced_rhs(params, y_temp)

    y_temp = [y_curr[i] + k3[i] * dt for i in range(DIM_REDUCED)]
    k4 = reduced_rhs(params, y_temp)

    return [y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0 for i in range(DIM_REDUCED)]


def integrator_rk4(y_curr, a, b, dt, n, stride, kneadings_start, kneadings_end):
    """Calculates kneadings during integration"""
    # n -- количество шагов интегрирования
    # stride -- через сколько шагов начинаем считать нидинги
    # first_derivative_curr, prev -- значения производных системы на текущем шаге и на предыдущем

    print(f'starting calculating with {y_curr} {a,b}')

    deriv_prev = 0
    deriv_curr = 0
    kneading_index = 0
    kneadings_weighted_sum = 0
    domain_num = 0

    w = 0  # ЭТИ ПАРАМЕТРЫ НУЖНО ПЕРЕДАВАТЬ ИЗ КОНФИГА
    r = 1  # ЭТИ ПАРАМЕТРЫ НУЖНО ПЕРЕДАВАТЬ ИЗ КОНФИГА

    params = [w, a, b, r]

    deriv_prev = avg_face_dist_deriv(params, y_curr)

    for i in range(1, n):

        for j in range(stride):
            y_curr = stepper_rk4(params, y_curr, dt)

        bary_coords = bary_expansion(y_curr)  # получаем барицентрические координаты точки
        domain_num = get_domain_num(bary_coords)  # получаем номер её подтетраэдра

        for k in range(DIM_REDUCED):
            if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                print('infinity')
                return InfinityError

        deriv_curr = avg_face_dist_deriv(params, y_curr)

        # проверяем, происходит ли max по расстоянию
        if deriv_prev > 0 > deriv_curr:

            if kneading_index >= kneadings_start:
                kneadings_weighted_sum += domain_num * 1 / (4.0 ** (-kneading_index + kneadings_end + 1))
            kneading_index += 1

        deriv_prev = deriv_curr

        if kneading_index > kneadings_end:
            print(kneadings_weighted_sum)
            return kneadings_weighted_sum

    print('did not end')
    return KneadingDoNotEndError


def sweep(
    inits,
    nones,
    alphas,
    betas,
    up_n,
    down_n,
    left_n,
    right_n,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    """Calls CUDA kernel and gets kneadings set back from GPU"""
    results = []

    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        init = [0.] * DIM_REDUCED

        if idx not in nones:
            init[0] = inits[idx * DIM_REDUCED + 0]
            init[1] = inits[idx * DIM_REDUCED + 1]
            init[2] = inits[idx * DIM_REDUCED + 2]

            result = integrator_rk4(init, alphas[idx], betas[idx], dt, n, stride, kneadings_start, kneadings_end)
        else:
            print(f'no init found for {init} {alphas[idx], betas[idx]}')
            result = NoInitFound

        results.append(result)

    return results


if __name__ == "__main__":
    dt = 0.01
    n = 50000
    stride = 1
    max_kneadings = 7

    inits_data = np.load(r'../src/system_analysis/inits.npz')

    inits = inits_data['inits']
    nones = inits_data['nones']
    alphas = inits_data['alphas']
    betas = inits_data['betas']
    up_n = inits_data['up_n']
    down_n = inits_data['down_n']
    left_n = inits_data['left_n']
    right_n = inits_data['right_n']

    kneadings_weighted_sum_set = sweep(
        inits,
        nones,
        alphas,
        betas,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        0,
        max_kneadings
    )

    np.savez(
        'kneadings_cpu_version.npz',
        kneadings=kneadings_weighted_sum_set
    )

    print("Results:")
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_number_system(kneading_weighted_sum, 4)

        print(f"a: {alphas[idx]:.6f}, "
              f"b: {betas[idx]:.6f} => "
              f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")
