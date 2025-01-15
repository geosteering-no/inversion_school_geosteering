import utils
import numpy as np

def get_log_with_noise(ref_log, log_offset_unit, delta_z, layer_depths, well_depths, noize_std=None, noize_rel_std=0.01):
    # convert the layer depth and the well depth to indexes
    # first compute relative positions
    rel_depths = well_depths - layer_depths
    rel_depths_ind = (rel_depths + log_offset_unit) / delta_z
    lateral_log = utils.eval_along_y_with_noize(ref_log, rel_depths_ind, noize_std=noize_std, noize_rel_std=noize_rel_std)
    return lateral_log

if __name__ == '__main__':
    import my_curve_data
    depths, ref_data = my_curve_data.get_data_series_default(plot=False)
    delta_z = depths[1] - depths[0]

    import my_trajectory_data
    geomodel_1d = my_trajectory_data.get_1d_geology_deafult(plot=False)

    lateral_well_shape = np.zeros(geomodel_1d.shape)

    lateral_log = get_log_with_noise(ref_log=ref_data,
                                     log_offset_unit=800.,
                                     delta_z=delta_z,
                                     layer_depths=geomodel_1d,
                                     well_depths=lateral_well_shape,
                                     noize_rel_std=0.04)

    import matplotlib.pyplot as plt
    plt.plot(lateral_log)
    plt.show()
