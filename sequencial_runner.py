# outside imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# utility imports
import heat_map
import utils

# solver imports
from submissions.solution_example import solve_sequencial as sergeys_solve_sequential
from submissions.tikhonov_solution_example import solve_sequencial as tikhonovs

if __name__ == "__main__":
    solver_dict = {
        'sergeys_test':{
            'solver': sergeys_solve_sequential,
            'z_prev': None,
            'prev_solution': None,
            'answer': None
        },
        'tikhonov_optimize_test':{
            'solver': tikhonovs
        }

    }

    import my_curve_data

    depths, ref_data = my_curve_data.get_data_series_default(plot=False)
    delta_z = depths[1] - depths[0]

    import my_trajectory_data

    geomodel_1d, coarse_geo_trend = my_trajectory_data.get_1d_geology_deafult(plot=False,
                                                                              get_trend_gradient=True)


    lateral_well_shape = np.zeros(geomodel_1d.shape)

    # constants for sequential inversion
    max_i = 7
    delta_x = 300
    from_ind = 3000
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

    for i in range(max_i):
        x1 = from_ind + delta_x * i
        x2 = from_ind + delta_x * (i + 1) - 1
        x = np.arange(x1, x2 + 1)
        log_segment = lateral_log[x1:x2 + 1]

        # initialize solutions with zeros
        for entry_key in solver_dict:
            # apply the solver following the API
            # solve_sequencial(range_x1_x2, lateral_log, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None)
            x, solution = solver_dict[entry_key]['solver'](
                [x1,x2], log_segment, ref_data, x1-1,
                prev_y=solver_dict[entry_key]['z_prev'],
                trend_gradient=coarse_geo_trend,
                prev_solution = solver_dict[entry_key]['prev_solution'],
                noize_level=noize_level
            )
            solver_dict[entry_key]['prev_solution'] = solution
            prev_y = solver_dict[entry_key]['z_prev'] = solution[-1]
            solver_dict[entry_key]['answer'][delta_x * i:delta_x * (i + 1)] = solution

    to_ind_extended = from_ind+max_i*delta_x
    image_part = heat_map.get_image_chunk(ref_data, lateral_log, from_ind, to_ind_extended, int(rel_depth_inds[from_ind]))
    plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    plt.plot(rel_depth_inds[from_ind:to_ind_extended]-rel_depth_inds[from_ind]+heat_map.CENTER_OFFSET, '--', color='black', alpha=0.5,
             label='True solution')
    for entry_key in solver_dict:
        # TODO improve visualization
        # compute misfit
        their_misfit = norm(rel_depth_inds[from_ind:to_ind_extended] - solver_dict[entry_key]['answer'], 1) / (to_ind_extended - from_ind)
        plt.plot(solver_dict[entry_key]['answer'] - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET,
                 color='black',
                 # alpha=0.5,
                 label=f"{entry_key} e={their_misfit:.2f}")
    plt.legend()

    plt.show()