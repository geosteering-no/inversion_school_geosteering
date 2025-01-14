import numpy as np
from trajectories_with_data_set import TrajectoryDatasetPlus
import my_curve_data

modes = 7
ref_len = 64
data_len = 16
prediction_len = 32
evaluaiton_len = prediction_len
mismatch_every = 2

test_data_set = TrajectoryDatasetPlus('test_data_trend.npz',
                                      fetch_data_method=my_curve_data.get_data_series_default,
                                      prediction_len=evaluaiton_len,
                                      data_len=data_len,
                                      ref_len=ref_len,
                                      mismatch_every=mismatch_every)

seed = 42 # default seed
rnd = np.random.default_rng(seed=seed)
test_data_set.shifts_for_data = rnd.integers(0, high=len(test_data_set.default_curve) - ref_len, size=(len(test_data_set)))
