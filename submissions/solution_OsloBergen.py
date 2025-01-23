import numpy as np
from scipy.optimize import curve_fit

import utils

def solve_sequencial(range_x1_x2, log_segment, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None,
                     noize_level=0.):
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]
    x = np.arange(x1, x2 + 1)

    trend_gradient = 0.04 #0.05
    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    print(y0_inp)
    y1_inp = y0_inp + trend_gradient * (x2 - prev_x)
    print(y1_inp)

    print("")
    def objective2(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1 
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.1, my_seed=455) #455
        return lateral_log_approx

    trend_gradient = 0.04 #0.05
    def objective3(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1 
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.015, my_seed=505) #455: 21.34, my_seed=505: 12.45
        return lateral_log_approx
    
    trend_gradient = 0.04 #0.05
    def objective4(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1 
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.015, my_seed=399) #399: 7.14
        return lateral_log_approx

    trend_gradient = 0.04 #0.05
    def objective5(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1 
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.01, my_seed=399) #0.01: 6.53
        return lateral_log_approx
    
    trend_gradient = 0.04 #0.05
    def objective6(x, y1, y2):
        b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1**2 
        # applying f(b(x))
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, b_line, noize_rel_std=0.009, my_seed=399) #0.01: 6.53
        return lateral_log_approx

    p_opt_2, p_cov_2 = curve_fit(objective5, x, log_segment, p0=[y0_inp, y1_inp],
                                 method='trf',
                                 loss='huber',
                                 bounds=(
                                     [y0_inp - 10., -np.inf],
                                     [y0_inp + 10., np.inf]
                                 )
                                 )
    opt_curve2 = (p_opt_2[1] - p_opt_2[0]) / (x2 - x1) * (x - x1) + p_opt_2[0]

    return x, opt_curve2
