import numpy as np
from scipy.optimize import curve_fit

import utils

def solve_sequencial(range_x1_x2, log_segment, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None,
                     noize_level=0.):
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]
    x = np.arange(x1, x2 + 1)

    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    y1_inp = y0_inp + trend_gradient * (x2 - prev_x)

    def objective2(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.0)
        return lateral_log_approx


    p_opt_2, p_cov_2 = curve_fit(objective2, x, log_segment, p0=[y0_inp, y1_inp],
                                 method='trf',
                                 loss='soft_l1',
                                 bounds=(
                                     [y0_inp - 15., -np.inf],
                                     [y0_inp + 15., np.inf]
                                 ), maxfev = 50000
                                 )
    opt_curve2 = (p_opt_2[1] - p_opt_2[0]) / (x2 - x1) * (x - x1) + p_opt_2[0]

    return x, opt_curve2
