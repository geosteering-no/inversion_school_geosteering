import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sympy.abc import alpha

import heat_map
import utils

if __name__ == "__main__":
    import my_curve_data
    depths, ref_data = my_curve_data.get_data_series_default(plot=False)
    delta_z = depths[1] - depths[0]

    import my_trajectory_data
    geomodel_1d = my_trajectory_data.get_1d_geology_deafult(plot=False)

    lateral_well_shape = np.zeros(geomodel_1d.shape)

    lateral_log, rel_depth_inds = heat_map.get_log_with_noise(ref_log=ref_data,
                                     log_offset_unit=905.,
                                     delta_z=delta_z,
                                     layer_depths=geomodel_1d,
                                     well_depths=lateral_well_shape,
                                     noize_rel_std=0.01,
                                     my_seed=0)

    delta_x = 300
    from_ind = 3000
    to_ind = from_ind+delta_x
    x1 = from_ind
    x2 = to_ind
    # center =

    def objective(x, a, b):
        line = a*x + b
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, line, noize_rel_std=0.0)
        return lateral_log_approx

    x = np.arange(from_ind, to_ind)
    log_segment = lateral_log[from_ind:to_ind]

    def objective2(x, y1, y2):
        line = (y2-y1)/(x2-x1)*(x-x1) + y1
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, line, noize_rel_std=0.0)
        return lateral_log_approx

    p_opt, p_cov = curve_fit(objective, x, log_segment, p0=[0, int(rel_depth_inds[from_ind])], method='trf', loss='arctan')
    opt_curve = p_opt[0] * x + p_opt[1]

    p_opt_2, p_cov_2 = curve_fit(objective2, x, log_segment, p0=[rel_depth_inds[from_ind], rel_depth_inds[from_ind]+5])
    opt_curve2 = (p_opt_2[1]-p_opt_2[0])/(x2-x1)*(x-x1) + p_opt_2[0]



    plt.figure()
    result = objective(x, p_opt[0], p_opt[1])
    result2 = objective2(x, p_opt_2[0], p_opt_2[1])
    plt.plot(result, '*-', label='Best fit', alpha=0.5)
    plt.plot(result2, label='Best fit 2', alpha=0.5)
    plt.plot(log_segment, 'v', label='True log', alpha=0.5)
    plt.legend()
    plt.figure()

    image_part = heat_map.get_image_chunk(ref_data, lateral_log, from_ind, to_ind, int(rel_depth_inds[from_ind]))
    plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    plt.plot(rel_depth_inds[from_ind:to_ind]-rel_depth_inds[from_ind]+heat_map.CENTER_OFFSET, '--', color='black', alpha=0.5)
    plt.plot(opt_curve-rel_depth_inds[from_ind]+heat_map.CENTER_OFFSET)
    plt.plot(opt_curve2 - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET)

    plt.show()
