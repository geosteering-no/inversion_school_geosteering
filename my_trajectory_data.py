import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_1d_geology_deafult(plot=False, get_trend_gradient=False):
    # original
    file_path = "trajectory_data/3da43bb8.csv"
    # file_path = "trajectory_data/c60a50fe.csv"
    # file_path = "trajectory_data/f954596a.csv"
    # # some variation
    # file_path = "trajectory_data/781ee7fe.csv"
    # file_path = "trajectory_data/f42b52f5.csv"

    df = pd.read_csv(file_path)
    # Define the new grid for VS_APPROX_adjusted with a fixed step of 1
    new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(), df['VS_APPROX_adjusted'].max() + 1, step=1)

    # Interpolate HORIZON_Z_adjusted values to the new grid
    new_horizon_z = np.interp(new_vs_grid, df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
    if plot:
        plt.plot(df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
        plt.plot(new_vs_grid, new_horizon_z)
        plt.show()
    if get_trend_gradient:
        # compute regional trend
        trend_distance = 300*15
        z2 = new_horizon_z[trend_distance]
        # return new_horizon_z, 15. / 300
        print(f"Trend delta z over {trend_distance} distance is {z2}. \n Gradient: {-z2/trend_distance}")
        return new_horizon_z, - z2 / trend_distance
    else:
        return new_horizon_z


if __name__ == "__main__":
    my_geology = get_1d_geology_deafult(plot=True)
    print(my_geology)


