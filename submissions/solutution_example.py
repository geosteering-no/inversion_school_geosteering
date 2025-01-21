import numpy as np
from scipy.optimize import curve_fit

import utils

def solve_sequencial(current_range_x1_x2, lateral_log, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None):
    x1 = current_range_x1_x2[0]
    x2 = current_range_x1_x2[1]
    x = np.arange(x1, x2 + 1)
    log_segment = lateral_log[x1:x2 + 1]

    y0_inp = prev_y[-1]
    y1_inp = y0_inp + (prev_y[-1] - prev_y[0]) / (prev_x[-1] - prev_x[0]) * (x2 - x1)

    def objective2(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.0)
        return lateral_log_approx


    p_opt_2, p_cov_2 = curve_fit(objective2, x, log_segment, p0=[y0_inp, y1_inp],
                                 method='trf',
                                 loss='soft_l1',
                                 bounds=(
                                     [y0_inp - 10., -np.inf],
                                     [y0_inp + 10., np.inf]
                                 )
                                 )
    opt_curve2 = (p_opt_2[1] - p_opt_2[0]) / (x2 - x1) * (x - x1) + p_opt_2[0]

    return x, opt_curve2
