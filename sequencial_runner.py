# outside imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import time

from torch.linalg import solve

# utility imports
import heat_map
import utils

# solver imports
from submissions.solution_example import solve_sequencial as sergeys_solve_sequential
from submissions.solution_OsloBergen import solve_sequencial as oslobergen_solve_sequential
from submissions.tikhonov_solution_example import solve_sequencial as tikhonovs
from submissions.yas_most_solution import solve_sequential as yas_kiwi_solver
from submissions.sb_1 import solve_sequencial as sb_1_solution
from submissions.solution_brute_magic_number import solve_sequencial as solution_brute_magic_number

if __name__ == "__main__":
    solver_dict = {
        'sergeys_test':{
            'solver': sergeys_solve_sequential,
        },
        'oslo_bergen':{
            'solver': oslobergen_solve_sequential,
        },
        'tikhonov_optimize_test':{
            'solver': tikhonovs
        },
        'yas_most_solution':{
            'solver': yas_kiwi_solver
        },
        'sb_1':{
            'solver': sb_1_solution
        },
        'brute_magic_number':{
            'solver': solution_brute_magic_number,
        },
    }

    import my_curve_data

    depths, ref_data = my_curve_data.get_data_series_default(plot=False)
    delta_z = depths[1] - depths[0]

    import my_trajectory_data

    geomodel_1d, coarse_geo_trend = my_trajectory_data.get_1d_geology_deafult(plot=False,
                                                                              get_trend_gradient=True)


    lateral_well_shape = np.zeros(geomodel_1d.shape)

    # constants for sequential inversion
    max_i = 10
    # max_i = 7
    delta_x = 300
    from_ind = 3000 - 950
    # from_ind = 3000
    to_ind = from_ind + delta_x
    x1 = from_ind
    x2 = to_ind - 1
    noize_level = 0.01

    # overwrite the gradient with more local one for simpler problem
    x0_trend = from_ind
    x1_trend = from_ind + delta_x * max_i
    y0_trend = geomodel_1d[x0_trend]
    y1_trend = geomodel_1d[x1_trend]
    trend_recomputed = - (y1_trend - y0_trend) / (x1_trend - x0_trend)
    # old hard-coded
    # coarse_geo_trend = 15. / 300
    coarse_geo_trend = trend_recomputed

    lateral_log, rel_depth_inds = heat_map.get_log_with_noise(ref_log=ref_data,
                                                              log_offset_unit=905.,
                                                              delta_z=delta_z,
                                                              layer_depths=geomodel_1d,
                                                              well_depths=lateral_well_shape,
                                                              noize_rel_std=noize_level,
                                                              my_seed=0)


    # initialize solutions with zeros
    for entry_key in solver_dict:
        solver_dict[entry_key]['prev_solution'] = None
        solver_dict[entry_key]['answer'] = np.zeros((delta_x * max_i))
        solver_dict[entry_key]['z_prev'] = rel_depth_inds[from_ind]
        solver_dict[entry_key]['misfit'] = 0.0
        solver_dict[entry_key]['total_time'] = 0.0

    to_ind_extended = from_ind+max_i*delta_x


    for i in range(max_i):
        x1 = from_ind + delta_x * i
        x2 = from_ind + delta_x * (i + 1) - 1
        x = np.arange(x1, x2 + 1)
        log_segment = lateral_log[x1:x2 + 1]

        # plotting

        plt.figure(i+10,(16,9))

        image_part = heat_map.get_image_chunk(ref_data, lateral_log, from_ind, to_ind_extended,
                                              int(rel_depth_inds[from_ind]))
        plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
        plt.plot(rel_depth_inds[from_ind:to_ind_extended] - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET,
                 '--', color='white', alpha=0.9,
                 label='True solution', linewidth=4.0)

        # end of plotting

        # initialize solutions with zeros
        for entry_key in solver_dict:
            # apply the solver following the API
            # solve_sequencial(range_x1_x2, lateral_log, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None)
            start_time = time.time()
            x, solution = solver_dict[entry_key]['solver'](
                [x1,x2], log_segment, ref_data, x1-1,
                prev_y=solver_dict[entry_key]['z_prev'],
                trend_gradient=coarse_geo_trend,
                prev_solution = solver_dict[entry_key]['prev_solution'],
                noize_level=noize_level
            )
            end_time = time.time()

            solver_dict[entry_key]['prev_solution'] = solution
            prev_y = solver_dict[entry_key]['z_prev'] = solution[-1]
            solver_dict[entry_key]['answer'][delta_x * i:delta_x * (i + 1)] = solution
            solver_dict[entry_key]['total_time'] += end_time - start_time

        best_key = None
        min_value = 100500*9000
        for entry_key in solver_dict:
            # TODO improve visualization
            # compute misfit
            full_dists = rel_depth_inds[from_ind:to_ind_extended] - solver_dict[entry_key]['answer']
            their_misfit = norm(full_dists[0:delta_x * (i + 1)], 1) / (delta_x * (i + 1))
            solver_dict[entry_key]['misfit'] = their_misfit
            if their_misfit < min_value:
                min_value = their_misfit
                best_key = entry_key

        # for the best key
        entry_key = best_key
        submission_part = solver_dict[best_key]['answer'][0:delta_x * (i + 1)]
        plt.plot(submission_part - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET,
                 color='black',
                 linewidth=3.0,
                 # alpha=0.8,
                 label=f"{best_key} e={solver_dict[best_key]['misfit']:.2f}   t={solver_dict[best_key]['total_time']:.1}s"
                 )

        for entry_key in solver_dict:
            if entry_key == best_key:
                continue
            submission_part = solver_dict[entry_key]['answer'][0:delta_x * (i + 1)]
            plt.plot(submission_part - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET,
                     color='black',
                     linewidth=2.0,
                     alpha=0.8,
                     label=f"{entry_key} e={solver_dict[entry_key]['misfit']:.2f}   t={solver_dict[entry_key]['total_time']:.1}s"
                     )
        plt.legend()
        plt.savefig(f'competition_results/result_step_{i}.png', dpi=600, bbox_inches='tight')
        # plt.show()

    exit(0)

    image_part = heat_map.get_image_chunk(ref_data, lateral_log, from_ind, to_ind_extended, int(rel_depth_inds[from_ind]))
    plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    for entry_key in solver_dict:
        # TODO improve visualization
        # compute misfit
        their_misfit = norm(rel_depth_inds[from_ind:to_ind_extended] - solver_dict[entry_key]['answer'], 1) / (to_ind_extended - from_ind)
        plt.plot(solver_dict[entry_key]['answer'] - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET,
                 color='black',
                 linewidth = 2.0,
                 # alpha=0.5,
                 # label=f"{entry_key} e={their_misfit:.2f}"
                 )

    # reasonable time: "5 minutes" - Øystein said as the boss

    plt.legend()


    # plt.show()

