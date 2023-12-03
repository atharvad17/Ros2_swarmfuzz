import pandas as pd
import numpy as np

def cal_summary(csv1, csv2, csv3):
    col_mat = pd.read_csv(csv1).to_numpy()
    dist_mat = pd.read_csv(csv2).to_numpy()

    col_flag = 0
    start_t = col_mat[0, 2]
    dur = col_mat[0, 3]

    # Check for collision
    col_idx = np.where(col_mat[:, 1] == 1)[0]
    if len(col_idx) > 0:
        col_flag = 1
        col_time = col_mat[col_idx.min(), 0]
        id_dist = np.where(dist_mat[:, 0] == col_time)[0]
        dist_obs_min = dist_mat[id_dist, 1].item()
    else:
        # Minimal VDO
        dist_min_idx = np.argmin(dist_mat[:, 1])
        col_time = dist_mat[dist_min_idx, 0]
        dist_obs_min = dist_mat[dist_min_idx, 1]

    sum_mat = np.array([start_t, dur, col_flag, col_time, dist_obs_min])
    pd.DataFrame([sum_mat]).to_csv(csv3, index=False, header=False, sep=',')
    
    # Delete input CSV files
    import os
    os.remove(csv1)
    os.remove(csv2)

# Example usage:
# csv1 = 'col_mat.csv'
# csv2 = 'dist_mat.csv'
# csv3 = 'sum_mat.csv'
# cal_summary(csv1, csv2, csv3)
