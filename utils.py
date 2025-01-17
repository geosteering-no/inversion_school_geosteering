import random
import scipy
import numpy as np

# random.seed(0)

def convert_to_image(reference_log, my_log):
    # reference is 1d numpy array
    # my_log is 1d numpy array
    # make reference to 2d "unsqueeze"
    ref2 = np.expand_dims(reference_log, axis=1)
    input_data_size = len(my_log)
    # make repeat for numpy
    ref_mat = np.repeat(ref2, input_data_size, axis=1)
    # make data to 2d "unsqueeze"
    data1 = np.expand_dims(my_log, axis=0)
    ref_data_size = len(reference_log)
    data_mat = np.repeat(data1, ref_data_size, axis=0)
    image = ref_mat - data_mat
    # image = image.unsqueeze(1)
    return image

def default_traj_to_index(self, traj, device):
    half = self.input_ref_size // 2
    index = traj * half + half
    index_trunc = torch.maximum(index, torch.ones(index.size(), device=device))
    index_trunc = torch.minimum(index_trunc,
                                torch.ones(index.size(), device=device) * (self.input_ref_size-2))
    return index_trunc

def evaluate_log(ref_data, output, my_arange, device):
    # todo refactor to remove self
    my_arange_uns = my_arange.unsqueeze(1)
    my_arange_uns = my_arange_uns.unsqueeze(2)
    #TODO fixme should be equal to the inversion size not prediciton size
    first_index = my_arange_uns.repeat(1, self.modes, self.input_data_size)
    expanded = self._expand(output)[:, :, 0:self.input_data_size]
    expanded_trunc = self.default_traj_to_index(expanded, device)
    indexes = (expanded_trunc + 0.5).long()
    i0s = (expanded_trunc).floor().detach()
    i1s = i0s + 1
    # indexes = torch.max(indexes, 0)
    # indexes = torch.min(indexes, self.input_ref_size-1)
    curves_values0 = ref_data[first_index, i0s.long()]
    curves_values1 = ref_data[first_index, i1s.long()]
    dists0 = expanded_trunc - i0s
    dists1 = i1s - expanded_trunc
    curves_values = dists1 * curves_values0 + dists0 * curves_values1
    return curves_values

def eval_along_y_with_noize(ref_data, ref_y, noize_std=None, noize_rel_std=0.01, my_seed=None):
    if my_seed is None:
        np.random.seed(random.randint(0, 100500))
    else:
        np.random.seed(seed=my_seed)
    my_inds = (ref_y + 0.5).astype(int)
    old_data = ref_data[my_inds]
    i0s = np.floor(ref_y).astype(int)
    i1s = i0s + 1
    # indexes = torch.max(indexes, 0)
    # indexes = torch.min(indexes, self.input_ref_size-1)
    curves_values0 = ref_data[i0s]
    curves_values1 = ref_data[i1s]
    dists0 = ref_y - i0s
    dists1 = i1s - ref_y
    if noize_std is None:
        min_data = np.min(ref_data)
        max_data = np.max(ref_data)
        data_range = max_data - min_data
        noize_std = data_range * noize_rel_std
        # print('Computed noize std ', noize_std)

    # correlated noise
    # x_for_corr = np.arange(curves_values0.size).reshape(curves_values0.size, 1) / curves_values0.size
    # dist = scipy.spatial.distance.pdist(x_for_corr)
    # dist = scipy.spatial.distance.squareform(dist)
    # correlation_scale = 1  # harf coded
    # cov = np.exp(-dist ** 2 / (2 * correlation_scale))
    # noise = np.random.multivariate_normal(0*curves_values0, cov)

    # fast correlated noise
    # correlation_scale = curves_values0.size // 8
    correlation_scale = 8
    dist = np.arange(-correlation_scale, correlation_scale)
    noise = np.random.normal(scale=noize_std, size=curves_values0.size)
    filter_kernel = np.exp(-dist ** 2 / (2 * correlation_scale))
    noise_correlated = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')

    # plt.figure()
    # plt.plot(noise)
    # plt.plot(noise_correlated)
    # plt.show()
    # exit()
    curves_values = dists1 * curves_values0 + dists0 * curves_values1 + noise_correlated
    curves_values = np.maximum(curves_values, 0)
    # todo add noize
    # my_inds = min(my_inds, self.ref_len-1)
    # my_inds = max(my_inds, 0)
    return curves_values

