import numpy as np
from numpy.linalg import norm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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

    max_i = 7
    delta_x = 300
    from_ind = 3000
    to_ind = from_ind+delta_x
    x1 = from_ind
    x2 = to_ind - 1
    # center =

    def objective(x, a, b):
        line = a*x + b
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, line, noize_rel_std=0.0)
        return lateral_log_approx

    x = np.arange(from_ind, to_ind)
    log_segment = lateral_log[from_ind:to_ind]



    p_opt, p_cov = curve_fit(objective, x, log_segment, p0=[0, int(rel_depth_inds[from_ind])], method='trf', loss='arctan')
    opt_curve = p_opt[0] * x + p_opt[1]

    y0_inp = rel_depth_inds[from_ind]
    y1_inp = rel_depth_inds[from_ind] + 15


    opt_curve_2_total = np.zeros((delta_x*max_i))

    for i in range(max_i):
        x1 = from_ind + delta_x * i
        x2 = from_ind + delta_x * (i+1) - 1
        x = np.arange(x1, x2+1)
        log_segment = lateral_log[x1:x2+1]
        prior_b_mean = (y1_inp - y0_inp) / (x2 - x1) * (x - x1) + y0_inp
        gamma = 0.009
        def objective3(y, x_var):
            y1 = y[0]
            y2 = y[1]
            x = x_var
            line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
            lateral_log_approx = utils.eval_along_y_with_noize(ref_data, line, noize_rel_std=0.0)
            distance_to_prior = line - prior_b_mean

            loss = norm(lateral_log_approx, 2) + gamma*norm(distance_to_prior, 2)
            return loss

        opt_result = minimize(objective3,
                                    x0 = [y0_inp, y1_inp],
                                    args=x
                                    )
        p_opt_2 = opt_result.x
        opt_curve2 = (p_opt_2[1]-p_opt_2[0])/(x2-x1)*(x-x1) + p_opt_2[0]
        opt_curve_2_total[delta_x*i:delta_x*(i+1)] = opt_curve2
        y0_inp = p_opt_2[1]
        y1_inp = y0_inp + 15


    to_ind_extended = from_ind+max_i*delta_x
    image_part = heat_map.get_image_chunk(ref_data, lateral_log, from_ind, to_ind_extended, int(rel_depth_inds[from_ind]))
    plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    plt.plot(rel_depth_inds[from_ind:to_ind_extended]-rel_depth_inds[from_ind]+heat_map.CENTER_OFFSET, '--', color='black', alpha=0.5)
    plt.plot(opt_curve-rel_depth_inds[from_ind]+heat_map.CENTER_OFFSET)
    plt.plot(opt_curve_2_total - rel_depth_inds[from_ind] + heat_map.CENTER_OFFSET)

    plt.show()
