import numpy as np

import utils
import multiprocessing as mp
from multiprocessing import current_process
from tqdm import tqdm
from operator import itemgetter

from tslearn.metrics import dtw_path
from scipy.spatial.distance import cdist
from itertools import product
import matplotlib.pyplot as plt

run_n = 0

def cool_figure(
    y1,
    y2,
    path
):
    y1 = np.array(y1).reshape((-1, 1))
    y2 = np.array(y2).reshape((-1, 1))
    sz1 = y1.shape[0]
    sz2 = y2.shape[0]

    plt.figure(1, figsize=(8, 8))

    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    height = 0.5*1
    width = 0.5*sz2/sz1
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(y1, y2)

    ax_gram.imshow(mat, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
                linewidth=3.)

    ax_s_x.plot(np.arange(sz2), y2, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz2 - 1))

    ax_s_y.plot(- y1, np.arange(sz1), "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz1 - 1))

    plt.tight_layout()
    plt.show()

previous_r2 = None

def solve_sequencial(
    range_x1_x2,
    log_segment,
    ref_data,
    prev_x,
    prev_y,
    trend_gradient,
    prev_solution = None,
    noize_level = 0.0
):
    # Load inputs
    x1 = range_x1_x2[0]
    x2 = range_x1_x2[1]

    # Initialise the previous ref_data range based on the previous gradient
    global previous_r2
    if previous_r2 is None:
        previous_r2 = x1 - 100

    global run_n
    run_n += 1
    print(
        "run_n: ", run_n, "\n",
        "range_x1_x2 [", len(range_x1_x2), "]: ", range_x1_x2, "\n",
        "ref_data [", len(ref_data), "]: ", ref_data, "\n",
        "trend_gradient: ", trend_gradient, "\n",
        "previous_r2: ", previous_r2, #"\n",
    )

    # Construct the entire x sample points within the considered range
    x = np.arange(x1, x2 + 1)

    # Brute force the combinations
    ranges = product(
        range(previous_r2 - 100, previous_r2 + 100 + 1),
        range(previous_r2 - 100, previous_r2 + 100 + 1 + len(x))
    )

    # Filter the allowed combinations
    ranges = filter((lambda r: 0 < r[0]), ranges)
    ranges = filter((lambda r: r[0] < r[1]), ranges)
    ranges = filter((lambda r: r[1] < len(ref_data)), ranges)
    ranges = filter((lambda r: r[1] - r[0] + 1 <= len(x)), ranges)
    ranges = filter((lambda r: r[1] - r[0] + 1 >= 100), ranges)
    ranges = list(ranges)

    if len(ranges) == 0:
        print("ERROR 1")
        exit()
    else:
        print(len(ranges))

    # Split ranges among threads
    n_cpus = mp.cpu_count()
    chunk_size = len(ranges) // n_cpus
    chunk_ranges = [
        slice(
            chunk_size * idx,
            chunk_size * (idx+1) if idx < (n_cpus-1) else None)
        for idx in range(n_cpus)]

    global compute_best_worker
    def compute_best_worker(
        slice
    ):
        current = current_process()
        this_ranges = ranges[slice]

        best_similarity = None
        best_path = None
        best_range = None

        with tqdm(total = len(this_ranges)) as progress_bar:
            for r in tqdm(this_ranges,
                desc = str(current.name), position=current._identity[0] - 1):

                path, similarity = dtw_path(
                    log_segment, ref_data[range(r[0], r[1]+1)],
                )
                if best_similarity is None or similarity < best_similarity:
                    best_range = r
                    best_similarity = similarity
                    best_path = path
            progress_bar.update(1)

        return (best_similarity, best_path, best_range)

    results = []
    with mp.Pool(processes = n_cpus, initializer = tqdm.set_lock,
        initargs = (tqdm.get_lock(),)) as pool:

        results = pool.imap_unordered(
            func      = compute_best_worker,
            iterable  = chunk_ranges,
            chunksize = 1)
        results = [result for result in results]

    similarity, path, range_r = min(results, key = itemgetter(0))
    previous_r2 = range_r[1]

    y0_inp = prev_y + trend_gradient * (x1 - prev_x)
    new_gradient = 0.15*(path[-1][1]+1) / (path[-1][0]+1)
    opt_curve2 = (new_gradient * (x - x1)) + y0_inp

    print(
        "range_r: [", range_r[1]-range_r[0]+1,"]", range_r, "\n",
        "new_gradient: ", new_gradient, "\n",
    #    "results: ", results, "\n",
    #    "similarity: ", similarity, "\n",
    #    "range: ", r, "\n",
    #    "path: [", len(path), "]", path, "\n",
    #    "y0_inp: ", y0_inp, "\n",
    #    "y1_inp: ", y1_inp, "\n",
    #    "log_segment [", len(log_segment), "]: ", log_segment, "\n",
    #    "prev_x [", "1", "]: ", prev_x, "\n",
    #    "prev_y [", "1", "]: ", prev_y, "\n",
    #    "trend_gradient [", "1", "]: ", trend_gradient, "\n",
    #    "prev_solution [", "1", "]: ", prev_solution, "\n",
    #    "noize_level [", "1", "]: ", noize_level "\n",
    )

    #cool_figure(log_segment, ref_data[range(range_r[0], range_r[1]+1)], path)

    return x, opt_curve2
