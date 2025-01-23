"""
Created on Wed Jan 22 17:21:38 2025

@author: 2925376
"""

import numpy as np
from scipy.optimize import curve_fit

import utils

def solve_sequential(range_x1_x2, log_segment, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None,
                     noize_level=0.):
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]
    x = np.arange(x1, x2 + 1)

    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    y1_inp = y0_inp + trend_gradient * (x2 - prev_x)

    def objective2(x, y1, y2):
        #b_line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        # applying f(b(x))
        
        a = (y2 - y1) / ((x2 - x1)**2)  # Coefficient for quadratic term
        b = -2 * a * x1  # Coefficient for linear term
        c = a * x1**2 + y1  # Constant term
        
        # Compute the quadratic polynomial value for the input x
        b_line = a * (x**2) + b * x + c
        
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
