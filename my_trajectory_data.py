import numpy as np
import pandas as pd

def get_1d_geology_deafult():
    file_path = "trajectory_data/3da43bb8.csv"
    df = pd.read_csv(file_path)
    # Define the new grid for VS_APPROX_adjusted with a fixed step of 1
    new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(), df['VS_APPROX_adjusted'].max() + 1, step=1)

    # Interpolate HORIZON_Z_adjusted values to the new grid
    new_horizon_z = np.interp(new_vs_grid, df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
    return new_horizon_z


if __name__ == "__main__":
    my_geology = get_1d_geology_deafult()
    print(my_geology)

    import matplotlib.pyplot as plt

    plt.plot(my_geology)
    plt.show()