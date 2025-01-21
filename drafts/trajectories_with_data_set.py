import matplotlib.pyplot as plt
import scipy
import scipy.signal

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class TrajectoryDatasetPlus(Dataset):
    def __init__(self, file_name_trend, fetch_data_method,
                 ref_len=400,
                 data_len=64,
                 prediction_len=96,
                 transform=None,
                 max_mismatch_magnitude_ft=15.,
                 mismatch_every=3,
                 noise=0.):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(file_name_trend, 'rb') as f:
            self.data = np.load(f)['data']
        self.transform = transform
        self.ref_len = ref_len
        self.data_len = data_len
        self.prediciton_len = prediction_len
        self.data_shape = self.data.shape
        self.default_curve = fetch_data_method()
        self.max_mismatch_magnitude_ft = max_mismatch_magnitude_ft
        self.mismatch_every = mismatch_every
        self.scale_from_ft = 2.
        self.noise = noise
        rng = np.random.default_rng(seed=0)
        self.shifts_for_data = rng.integers(0, high=len(self.default_curve)-ref_len, size=(len(self)))
        # these shifts/trnasforms are to promote divesity
        # self.traj_shift = rng.integers(ref_len // 4, high=(ref_len // 4) * 3, size=(len(self)))
        # self.traj_mult = rng.integers(1, high=2, size=(len(self)))
        # self.traj_neg = rng.integers(0, high=2, size=(len(self)))
        print('modulo', self._get_modulo())


    def __len__(self):
        modulo = self._get_modulo()
        return self.data_shape[0] * modulo

    def _get_modulo(self):
        # modulo = 1
        modulo = self.data_shape[1] - self.prediciton_len - 1
        return modulo

    def _transform_traj(self, traj, shift=0, mult=1.0, neg=0):
        if neg != 0:
            mult *= -1
        transformed_traj = traj * mult + shift
        transformed_traj[transformed_traj > self.ref_len - 1] = self.ref_len - 1
        transformed_traj[transformed_traj < 0] = 0
        return transformed_traj

    def _eval_along_y_with_noize(self, ref_data, ref_y, noize_std=None, noize_rel_std=0.01):
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
        # todo add noize
        # my_inds = min(my_inds, self.ref_len-1)
        # my_inds = max(my_inds, 0)
        return curves_values

    def _eval_along_y(self, ref_data, ref_y):
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
        curves_values = dists1 * curves_values0 + dists0 * curves_values1
        # todo add noize
        # my_inds = min(my_inds, self.ref_len-1)
        # my_inds = max(my_inds, 0)
        return curves_values

    def _get_single_item(self, idx):
        modulo = self._get_modulo()
        first_ind = idx // modulo
        second_ind = idx % modulo

        # getting the reference data (typelog)
        shift = self.shifts_for_data[idx]
        ref_data_vector = self.default_curve[shift:shift+self.ref_len]

        # getting the trajectory
        trajectory = self.data[first_ind, second_ind:second_ind+self.prediciton_len, 0:2]
        traj_y = trajectory[:, 1]
        output_traj = traj_y
        shift = self.ref_len // 2
        # we will not move the origin to simulate mistakes
        if idx % self.mismatch_every != 0:
            shift += -traj_y[0] * self.scale_from_ft
        #     output_traj_2 = output_traj - traj_y[0]
        # else:
        #     output_traj_2 = output_traj
        transformed_traj = self._transform_traj(traj_y,
                                                shift=shift,
                                                mult=self.scale_from_ft)

        # getting data for the part of trajectory
        transformed_traj_part = transformed_traj[0:self.data_len]
        my_data_vector = self._eval_along_y_with_noize(ref_data=ref_data_vector,
                                                       ref_y=transformed_traj_part,
                                                       noize_rel_std=self.noise)

        output_traj_2 = self.traj_from_index(transformed_traj)

        return ref_data_vector, my_data_vector, output_traj_2

    def traj_from_index(self, traj_ind):
        half = self.ref_len // 2
        output_traj_2 = (traj_ind - half) / half
        return output_traj_2

    def index_from_traj(self, traj):
        half = self.ref_len // 2
        traj_ind = traj * half + half
        return traj_ind

    def __getitem__(self, idx):
        """

        :param idx: list of indexes
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            return self._get_single_item(idx)

        raise Exception('asking for multi index not implemented properly')

        my_inputs = []
        my_outputs = []

        for i in idx:
            input, output = self._get_single_item(i)
            my_inputs.append(input)
            my_outputs.append(output)

        sample = (my_inputs, my_outputs)

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    import my_curve_data
    data_len = 64
    dataset = TrajectoryDatasetPlus('test_data.npz',
                                    my_curve_data.get_data_series_default,
                                    data_len=data_len)
    for i, data in enumerate(dataset):
        print(i)
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[2])
        if i > 5:
            break

    dataloader = DataLoader(dataset, batch_size=2, drop_last=True,
                            shuffle=True, num_workers=8, pin_memory=True,
                            generator=torch.Generator().manual_seed(42))

    for i, data in enumerate(dataloader, 0):
        inputs1, inputs2, outputs = data
        plt.figure()
        curve = dataset.index_from_traj(outputs.numpy()[0,:])
        curve_arange = np.arange(0, len(curve)) * 100. / data_len
        plt.plot(curve_arange, curve)
        plt.plot(curve_arange[0:data_len], curve[0:data_len])
        plt.gca().axis('scaled')
        plt.figure()
        log2 = inputs2.numpy()[0,:]
        plt.plot(log2)
        plt.plot(log2)
        diff = max(log2) - min(log2)
        delta = max(diff / 10., 1e-3)
        plt.ylim(min(log2) - delta, max(log2) + delta)
        plt.figure()
        log1 = inputs1.numpy()[0,:]
        plt.plot(log1)
        curve = curve[0:data_len]
        print(int(min(curve)+.5), int(max(curve)+.5))
        subset = np.arange(int(min(curve)+.5), int(max(curve)+.5)+1)
        plt.plot(subset, log1[subset])
        plt.xlim(int(min(curve)) - 5, int(max(curve)) + 5)
        plt.ylim(min(log2) - delta, max(log2) + delta)
        plt.show()
        if i > 10:
            break



