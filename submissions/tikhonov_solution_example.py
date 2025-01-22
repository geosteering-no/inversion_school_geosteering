import numpy as np
from scipy.optimize import minimize

import utils

def solve_sequencial(range_x1_x2, log_segment, ref_data, prev_x, prev_y, trend_gradient, prev_solution=None,
                     noize_level=0.):
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]
    x = np.arange(x1, x2 + 1)

    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    y1_inp = y0_inp + trend_gradient * (x2 - prev_x)

    prior_b_mean = (y1_inp - y0_inp) / (x2 - x1) * (x - x1) + y0_inp
    gamma = 0.009

    def objective3(y, x_var):
        y1 = y[0]
        y2 = y[1]
        x = x_var
        line = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        lateral_log_approx = utils.eval_along_y_with_noize(ref_data, line, noize_rel_std=0.0)
        distance_to_prior = line - prior_b_mean

        loss = np.sum(lateral_log_approx**2 + gamma * distance_to_prior**2)
        return loss

    opt_result = minimize(objective3,
                          x0=[y0_inp, y1_inp],
                          args=x
                          )
    p_opt_2 = opt_result.x
    opt_curve2 = (p_opt_2[1] - p_opt_2[0]) / (x2 - x1) * (x - x1) + p_opt_2[0]

    return x, opt_curve2
