import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_1d_geology_deafult(plot=False):
    # original
    file_path = "trajectory_data/3da43bb8.csv"
    # # faulted
    # file_path = "trajectory_data/8b440fd7.csv"
    # file_path = "trajectory_data/8f434a7e.csv"
    # # some variation
    # file_path = "trajectory_data/8f4560a1.csv"
    # file_path = "trajectory_data/9ac9aa08.csv"

    df = pd.read_csv(file_path)
    # Define the new grid for VS_APPROX_adjusted with a fixed step of 1
    new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(), df['VS_APPROX_adjusted'].max() + 1, step=1)

    # Interpolate HORIZON_Z_adjusted values to the new grid
    new_horizon_z = np.interp(new_vs_grid, df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
    if plot:
        plt.plot(df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
        plt.plot(new_vs_grid, new_horizon_z)
        plt.show()
    return new_horizon_z


if __name__ == "__main__":
    my_geology = get_1d_geology_deafult(plot=True)
    print(my_geology)


