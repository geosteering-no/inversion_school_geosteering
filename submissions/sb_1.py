import numpy as np
from scipy.optimize import minimize, curve_fit, least_squares

import utils

def solve_sequencial(range_x1_x2 # index range 0:300 i.e.
                         , log_segment #data batch
                         , ref_data
                         , prev_x
                         , prev_y
                         , trend_gradient
                         , prev_solution=None,
                     noize_level=0.):
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]
    x = np.arange(x1, x2 + 1)

    # Initial guess for the optimization based on previous values and trend gradient
    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    y1_inp = y0_inp + trend_gradient * (x2 - prev_x)

    # Objective function to minimize the difference between the approximated log and the actual log segment
    def objective2(params):
        y1, y2 = params
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.0)
        return np.sum((lateral_log_approx - log_segment) ** 2)

    # Minimize the objective function to find the optimal y1 and y2
    result = minimize(objective2, [y0_inp, y1_inp], bounds=[(y0_inp - 0.1, y0_inp + 0.1), (-np.inf, np.inf)])
    y1_opt, y2_opt = result.x

    # Compute the optimized curve based on the optimal y1 and y2
    opt_curve2 = (y2_opt - y1_opt) / (x2 - x1) * (x - x1) + y1_opt

    def objective3(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.0)
        return lateral_log_approx

    p_opt_2, p_cov_2 = curve_fit(objective3, x, log_segment, p0=[y0_inp, y1_inp],
                                 method='dogbox',  # options: 'lm', 'trf', 'dogbox'
                                 loss='linear',  # options: 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'
                                 bounds=(
                                     [y0_inp - 0.1, -np.inf],
                                     [y0_inp + 0.1, np.inf]
                                 ), maxfev=50000
                                 )
    opt_curve3 = (p_opt_2[1] - p_opt_2[0]) / (x2 - x1) * (x - x1) + p_opt_2[0]

    # Least squares objective function
    def objective4(params):
        y1, y2 = params
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.0)
        return lateral_log_approx - log_segment

    result_ls = least_squares(objective4, [y0_inp, y1_inp], bounds=([y0_inp - 0.1, -np.inf], [y0_inp + 0.1, np.inf]))
    y1_ls, y2_ls = result_ls.x
    opt_curve4 = (y2_ls - y1_ls) / (x2 - x1) * (x - x1) + y1_ls

    opt_curve = 0.40 * opt_curve2 + 0.40 * opt_curve3 + 0.20 * opt_curve4

    return x, opt_curve
