import numpy as np
import pprint
import matplotlib.pyplot as plt

from lib.computation_template.workers_utils import register, makeFinalOutname

import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so
from src.system_analysis.get_inits import continue_equilibrium, get_saddle_foci_grid, find_inits_for_equilibrium_grid, generate_parameters
from src.mapping.convert import decimal_to_number_system
from src.cuda_sweep.sweep_fbpo import sweep
from src.mapping.plot_kneadings import plot_mode_map, set_random_color_map

registry = {
    "worker": {},
    "init": {},
    "post": {}
}


@register(registry, 'init', 'kneadings_fbpo')
def init_kneadings_fbpo(config, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']

    param_to_index = config['misc']['param_to_index']
    start_eq = config['misc']['start_eq']

    grid_dict = config['grid']
    up_n = grid_dict['second']['up_n']
    up_step = grid_dict['second']['up_step']
    down_n = grid_dict['second']['down_n']
    down_step = grid_dict['second']['down_step']
    left_n = grid_dict['first']['left_n']
    left_step = grid_dict['first']['left_step']
    right_n = grid_dict['first']['right_n']
    right_step = grid_dict['first']['right_step']

    init_res_name = config['misc']['init_res']
    if init_res_name != "ignore":
        init_res = np.load("./input/" + init_res_name + ".npz")
        inits = init_res['inits']
        nones = init_res['nones']
        params_x = init_res['params_x']
        params_y = init_res['params_y']
        if len(inits) != (3 * (left_n + right_n + 1) * (up_n + down_n + 1)):
            raise Exception("Saved initial conditions array differs from entered grid params!")
        if not np.array_equal(start_eq, init_res['start_eq']):
            raise Exception("Start equilibrium of saved initial conditions array ")
    else:
        start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)
        reduced_rhs_wrapper = start_sys.getReducedSystem
        reduced_jac_wrapper = start_sys.getReducedSystemJac
        get_params = start_sys.getParams
        set_params = start_sys.setParams

        if start_eq is not None:
            eq_grid = continue_equilibrium(reduced_rhs_wrapper, reduced_jac_wrapper, get_params, set_params,
                                           param_to_index, 'a', 'b',
                                           start_eq, up_n, down_n, left_n, right_n,
                                           up_step, down_step, left_step, right_step)
            sf_grid = get_saddle_foci_grid(eq_grid, up_n, down_n, left_n, right_n, sf.STD_PRECISION)
            inits, nones = find_inits_for_equilibrium_grid(sf_grid, 3, up_n, down_n, left_n, right_n, sf.STD_PRECISION)
            params_x, params_y = generate_parameters((a, b), up_n, down_n, left_n, right_n,
                                                up_step, down_step, left_step, right_step)
        else:
            raise Exception("Start saddle-focus was not found!")

    return {'inits': inits, 'nones': nones, 'params_x': params_x, 'params_y': params_y, 'targetDir': 'output'}


@register(registry, 'worker', 'kneadings_fbpo')
def worker_kneadings_fbpo(config, initResult, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    def_params = [w, a, b, r]

    param_to_index = config['misc']['param_to_index']

    grid_dict = config['grid']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']
    param_x_name = grid_dict['first']['name']
    param_y_name = grid_dict['second']['name']

    kneadings_dict = config['kneadings_fbpo']
    dt = kneadings_dict['dt']
    n = kneadings_dict['n']
    stride = kneadings_dict['stride']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    kneadings_records = pprint.pformat(config) + "\n\n"

    kneadings_weighted_sum_set = sweep(
        inits,
        nones,
        params_x,
        params_y,
        def_params,
        param_to_index,
        param_x_name,
        param_y_name,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end
    )

    # print("Results:")
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_number_system(kneading_weighted_sum, 4)

        # print(f"a: {params_x[idx]:.15f}, "
        #       f"b: {params_y[idx]:.15f} => "
        #       f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")
        kneadings_records = (kneadings_records + f"a: {params_x[idx]:.15f}, "
                                                 f"b: {params_y[idx]:.15f} => "
                                                 f"{kneading_symbolic} (Raw: {kneading_weighted_sum})\n")

    return {'kneadings_weighted_sum_set': kneadings_weighted_sum_set, 'kneadings_records': kneadings_records}


@register(registry, 'post', 'kneadings_fbpo')
def post_kneadings_fbpo(config, initResult, workerResult, grid, startTime):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']

    plot_params_dict = config['misc']['plot_params']
    font_size = plot_params_dict['font_size']
    init_res_name = config['misc']['init_res']

    grid_dict = config['grid']
    up_n = grid_dict['second']['up_n']
    up_step = grid_dict['second']['up_step']
    down_n = grid_dict['second']['down_n']
    down_step = grid_dict['second']['down_step']
    left_n = grid_dict['first']['left_n']
    left_step = grid_dict['first']['left_step']
    right_n = grid_dict['first']['right_n']
    right_step = grid_dict['first']['right_step']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    start_eq = config['misc']['start_eq']

    kneadings_weighted_sum_set = workerResult['kneadings_weighted_sum_set']
    kneadings_records = workerResult['kneadings_records']

    param_x_caption = f"{grid_dict['first']['caption']}"
    param_x_count = left_n + right_n + 1
    param_x_start = a - left_n * left_step
    param_x_end = a + right_n * right_step
    param_y_caption = f"{grid_dict['second']['caption']}"
    param_y_count = up_n + down_n + 1
    param_y_start = b - down_n * down_step
    param_y_end = b + up_n * up_step

    plot_mode_map(kneadings_weighted_sum_set, set_random_color_map, param_x_caption, param_y_caption,
                  param_x_start, param_x_end, param_x_count, param_y_start, param_y_end, param_y_count,
                  font_size)
    plt.title(fr'$\omega = {w}$, $r = {r}$', fontsize=font_size)

    if init_res_name == "ignore":
        npz_outname = makeFinalOutname(config, initResult, "npz", startTime)
        np.savez(
            npz_outname,
            inits=inits,
            nones=nones,
            params_x=params_x,
            params_y=params_y,
            start_eq=start_eq
        )
        print("Init stage results successfully saved")

    npy_outname = makeFinalOutname(config, initResult, "npy", startTime)
    np.save(npy_outname, kneadings_weighted_sum_set)
    print("Kneadings set successfully saved")

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(f'{txt_outname}', 'w') as txt_output:
        txt_output.write(kneadings_records)
    print("Kneadings records successfully saved")

    img_extension = config['output']['imageExtension']
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    plt.savefig(plot_outname, dpi=600, bbox_inches='tight')
    plt.close()
    print("Mode map successfully saved")