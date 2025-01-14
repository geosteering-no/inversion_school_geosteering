import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


def get_data_series_default(plot=False, return_plot_data=False):
    start_tvd = 10000
    finish_tvd = 12000
    sample_ind = 1800

    data_frame = pd.read_csv('ref_data/gr.csv', index_col='md')
    data_frame_reduced = data_frame.loc[start_tvd:finish_tvd]
    my_arange = np.arange(start_tvd, finish_tvd+1e-5, 0.5) - start_tvd
    data_np = data_frame_reduced.to_numpy()

    # data_np = np.log10(data_np)
    print(data_np[sample_ind,:])
    scaler = MinMaxScaler()
    scaler.fit(data_np)
    print('min', scaler.data_min_)
    print('max', scaler.data_max_)
    data_np = scaler.transform(data_np)
    print(data_np[sample_ind, :])


    # print(data_np.shape)

    ref_data = data_np[:, 0]

    if plot:
        # todo save log plot
        plt.plot(my_arange, data_np)
        # plt.yscale('log')
        plt.title('Scaled gamma-ray log')
        plt.xlabel('relative TVD, ft')
        plt.tight_layout()
        plt.savefig('figs/gamma-ray.png', dpi=600)
        plt.savefig('figs/gamma-ray.pdf')
        plt.show()
    if return_plot_data:
        return (ref_data, my_arange, data_np)

    return ref_data


if __name__ == '__main__':
    get_data_series_default(plot=True)

