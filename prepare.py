import os
import numpy as np
import pandas as pd

def prepare(nb, c_obs, pos_csv, dist_csv, dire_csv, col_csv, seed):
    current_folder = os.getcwd()
    root_f = os.path.join(current_folder, '../')
    param_sim_path = os.path.join(current_folder, '../parameters/param_sim.py')
    
    # Load parameters from param_sim.py
    exec(open(param_sim_path).read())

    # In dire_csv, "1" means the swarm member passes the obstacle from the right;
    # "-1" means from the left
    r = 1
    l = -1
    
    # CSV files to store the properties of different types of missions;
    # Read if there are previous files storing the mission
    proper_csv = os.path.join(root_f, 'fuzz/tmp_files/mission_property.csv')
    if os.path.isfile(proper_csv):
        proper_mat = pd.read_csv(proper_csv).to_numpy()
    else:
        proper_mat = np.zeros((1, 3))
    
    new_proper = np.zeros((1, 3))  # used to record the property of the current mission [seed, collision, same_direction]
    new_proper[0, 0] = seed

    # Identify if there are collisions even without attack
    col_mat = pd.read_csv(col_csv).to_numpy()
    col_idx = np.where(col_mat[:, 1] == 1)[0]
    if col_idx.size > 0:
        new_proper[0, 1] = 1
    
    # Direction of drones when passing the obstacle - For building the graph
    pos_mat = pd.read_csv(pos_csv).to_numpy()
    final_dire = np.zeros(nb)
    early_idx = int(p_sim.end_time * 100)  # the time before the victim drone passes the obstacle
    
    for i in range(nb):
        # find the first point where drone reaches the same x as the obstacle
        idx = np.argmax(pos_mat[:, 2 * i] > c_obs[0])
        if np.isnan(idx):
            idx = int(p_sim.end_time * 100)
        early_idx = min(idx, early_idx)
        
        if pos_mat[idx, 2 * i + 1] > c_obs[1]:
            final_dire[i] = r
        else:
            final_dire[i] = l
    
    # Identify whether the passing direction of all drones is the same
    if np.abs(np.sum(final_dire)) == nb:
        # if the passing direction of all drones is the same, won't consider
        new_proper[0, 2] = 1
        print("***** The passing direction of all drones is the same, we'd better drop this. *****")
        return -1, -1, -1, np.zeros(nb)
    
    # Drop the collision missions
    if new_proper[0, 1] == 1:
        # if collisions occur without attack, won't consider
        print("***** Collisions occur without attack, won't consider. *****")
        proper_mat = update_proper_mat(proper_mat, new_proper)
        return -1, -1, -1, np.zeros(nb)
    
    # Initial attack start time - minimal sum - For fuzzing
    # Find out the time before the victim drone passes the obstacle (i.e., early_idx)
    rows = early_idx
    dist_mat = np.zeros((rows, 2))
    
    for i in range(rows):
        pos_row = pos_mat[i, 2:].reshape((nb, 2)).T  # each drone's location at time i
        dist_row = np.sum(np.array([np.linalg.norm(pos_row[:, j] - pos_row[:, k]) for j in range(nb) for k in range(j + 1, nb)]))
        dist_mat[i, :] = [pos_mat[i, 0], dist_row]
    
    start_idx = np.argmin(dist_mat[:, 1])  # find the time where the distance sum is the smallest
    start_t = max(pos_mat[start_idx, 0], 0.01)  # for gradient descent
    
    # Drones' location at the initial attack start time - For building the graph
    row_id = int(start_t * 100)
    pos_att = pos_mat[row_id, 1:(2 * nb + 1)].reshape((2, nb))
    
    # Min dist between the drone and the obstacle in the no-attack mission - For seed scheduling
    dist_obs = np.zeros(nb)
    dist_all = pd.read_csv(dist_csv).to_numpy()
    
    for i in range(nb):
        idx = np.where(dist_all[:, 1] == i + 1)[0]
        dist_obs[i] = np.min(dist_all[idx, 2])
    
    # Write the mission type into csv files, delete zero and repetitive number
    proper_mat = update_proper_mat(proper_mat, new_proper)
    
    return start_t, pos_att, dist_obs, final_dire


def update_proper_mat(proper_mat, new_proper):
    # find the property recording the same mission
    same_mission_idx = np.where(proper_mat[:, 0] == new_proper[0, 0])[0]
    
    if same_mission_idx.size > 0:
        proper_mat[same_mission_idx[0], :] = new_proper
        while len(same_mission_idx) > 1:
            proper_mat = np.delete(proper_mat, same_mission_idx[1], axis=0)
            same_mission_idx = np.where(proper_mat[:, 0] == new_proper[0, 0])[0]
    else:
        proper_mat = np.concatenate((proper_mat, new_proper), axis=0)
    
    # delete any zero rows
    proper_mat = proper_mat[~np.all(proper_mat == 0, axis=1)]
    
    # save to CSV
    pd.DataFrame(proper_mat, columns=['seed', 'collision', 'same_direction']).to_csv('proper_mat.csv', index=False)
    
    return proper_mat
