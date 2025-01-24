import utils
import numpy as np
import matplotlib.pyplot as plt

CENTER_OFFSET = 160

def get_log_with_noise(ref_log, log_offset_unit, delta_z, layer_depths, well_depths, noize_std=None, noize_rel_std=0.01, my_seed=None):
    # convert the layer depth and the well depth to indexes
    # first compute relative positions
    rel_depths = well_depths - layer_depths
    rel_depths_ind = (rel_depths + log_offset_unit) / delta_z
    lateral_log = utils.eval_along_y_with_noize(ref_log, rel_depths_ind, noize_std=noize_std, noize_rel_std=noize_rel_std, my_seed=my_seed)
    return lateral_log, rel_depths_ind

def convert_to_image(ref_log, lateral_log):
    return utils.convert_to_image(ref_log, lateral_log)

def plot_with_panels(ref_log, lateral_log, curve):
    image = convert_to_image(ref_log, lateral_log)
    # need to have a large image and two panels to the sides
    # the bottom panel shows the lateral log (same coordinates as the image)
    # the right paneel shows the reference log (same coordinates as the image)
    total_figs = 3
    # ax_drawing = plt.subplot2grid((4, total_figs * 5), (0, 2), colspan=total_figs * 3)
    ax_main = plt.subplot2grid((4, total_figs * 5), (1, 2), colspan=total_figs * 3, rowspan=2)
    # hide x-axis numbers
    ax_main.xaxis.set_visible(False)
    # ax_colorbar = plt.subplot2grid((4, total_figs * 5), (1, 0), rowspan=1)
    ax_cur_log = plt.subplot2grid((4, total_figs * 5), (3, 2), colspan=total_figs * 3,
                                  sharex=ax_main)
    ax_offset_log = plt.subplot2grid((4, total_figs * 5), (1, total_figs * 4),
                                     rowspan=2, colspan=2 * total_figs // 3,
                                     sharey=ax_main
                                     )
    # hide the y-axis numbers
    ax_offset_log.yaxis.set_visible(False)

    ax_geology = plt.subplot2grid((4, total_figs * 5), (0, total_figs * 4),
                                     rowspan=1, colspan=2 * total_figs // 3,
                                     # sharey=ax_drawing
                                     )

    my_shape = image.shape

    y_coord = np.arange(my_shape[0])

    # set values interval for the colorbar in imshow
    cax = ax_main.imshow(image, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    plt.colorbar(cax, cax=ax_geology)
    ax_main.plot(curve, '--', color='black', alpha=0.5)

    ax_cur_log.plot(lateral_log)

    ax_offset_log.plot(ref_log, y_coord)
    plt.show()

def get_image_chunk(ref_data, lateral_log, from_ind, to_ind, center):
    image = convert_to_image(ref_data, lateral_log)

    return image[center-CENTER_OFFSET:center+CENTER_OFFSET, from_ind:to_ind]


if __name__ == '__main__':
    import my_curve_data
    depths, ref_data = my_curve_data.get_data_series_default(plot=False)
    delta_z = depths[1] - depths[0]

    import my_trajectory_data
    geomodel_1d = my_trajectory_data.get_1d_geology_deafult(plot=False)

    lateral_well_shape = np.zeros(geomodel_1d.shape)

    lateral_log, rel_depth_inds = get_log_with_noise(ref_log=ref_data,
                                     log_offset_unit=1000.,
                                     delta_z=delta_z,
                                     layer_depths=geomodel_1d,
                                     well_depths=lateral_well_shape,
                                     noize_rel_std=0.01)

    from_ind = 3000
    to_ind = from_ind+500
    image_part = get_image_chunk(ref_data, lateral_log, from_ind, to_ind, int(rel_depth_inds[from_ind]))
    plt.imshow(image_part, aspect='auto', vmin=-0.1, vmax=0.1, cmap='bwr')
    plt.plot(rel_depth_inds[from_ind:to_ind]-rel_depth_inds[from_ind]+CENTER_OFFSET, '--', color='black', alpha=0.5)
    plt.show()


    # plot_with_panels(ref_data, lateral_log, rel_depth_inds)
