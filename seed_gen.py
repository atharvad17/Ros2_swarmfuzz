import os
import numpy as np
import pandas as pd
import networkx as nx

def seed_gen(nb, start_t, pos_att, dist_obs, final_dire, seedpool):
    # Initialization
    nb_v = 2  # number of victim drones selected for seed scheduling
    nb_t = 2  # number of target drones selected for seed scheduling
    dist_thre = 5  # threshold for the distance difference between top nb_v victim drones
    pool = np.zeros((nb_v * nb_t, 4))  # initialize the seedpool
    flag = -1

    # Construct swarm vulnerability graph
    # 1. Identify the target direction of the victim drones when under attacks
    # The attack goal for the direction of victim drones - opposite with the drones' direction in the no-attack scenario
    goal = -final_dire

    # 2. Decide the initial victim drones
    # Choose the top nb_v (e.g., 2) drones with smallest distance as the initial victim drones
    dist_vic_tmp, vic_id_tmp = np.sort(dist_obs)[:nb_v], np.argsort(dist_obs)[:nb_v]
    # If the distance difference between the top nb_v drones is > dist_thre, then drop the drone with larger distance
    if dist_vic_tmp[1] > (dist_vic_tmp[0] + dist_thre):
        vic_id_tmp = [vic_id_tmp[0]]
        nb_v = 1

    # 3. Build the graph and compute the page rank centrality
    # Consider 2 scenarios:
    # dev = 1 - the target drone deviates to its right side.
    # dev = -1 - the target drone deviates to its left side.
    dev = [1, -1]
    A = np.zeros((2, nb, nb))  # graphs in 2 scenarios
    flags = np.zeros(2)  # validity for graph in 2 scenarios
    pgrank_tar = np.zeros((2, nb))  # pagerank centrality for target drones in 2 scenarios
    pgrank_vic = np.zeros((2, nb))  # pagerank centrality for victim drones in 2 scenarios

    for i in range(len(dev)):  # for each deviation direction
        A[i, :, :], flags[i] = build_graph(dev[i], goal, pos_att, nb)  # build the graph A
        # If the graph exists, compute the pagerank centrality
        if flags[i] == 1:
            # Graph for the influential nodes (target drones)
            G_tar = nx.DiGraph(A[i, :, :])
            pgrank_tar[i, :] = np.array(list(nx.pagerank(G_tar, weight='weight').values()))
            # Graph for the nodes being influenced (victim drones)
            G_vic = nx.DiGraph(A[i, :, :].T)
            pgrank_vic[i, :] = np.array(list(nx.pagerank(G_vic, weight='weight').values()))

    # Seed scheduling
    valid_idx = np.where(flags == 1)[0]
    # If both left and right deviation help with the attack goal
    if len(valid_idx) == 2:
        flag = 1
        # Select the most promising victim drone, i.e., closest to the obstacle.
        # Then, compare the pg_rank centrality of this victim drone in two scenarios,
        # to decide what kind of deviation should be prioritized.
        for i in range(nb_v):
            idx = vic_id_tmp[i]  # index of the initial victim drones
            tmp_pg_ranks_tar1 = pgrank_tar[0, :].copy()
            tmp_pg_ranks_tar2 = pgrank_tar[1, :].copy()
            tmp_pg_ranks_tar1[idx] = -1
            tmp_pg_ranks_tar2[idx] = -1
            if pgrank_vic[0, idx] > pgrank_vic[1, idx]:  # right deviation is more influential to this victim drone
                target_id = np.argsort(tmp_pg_ranks_tar1)[-nb_t:]  # thus, select 2 target drones with the highest score with right deviation
                f = [1, 1]
            elif pgrank_vic[0, idx] < pgrank_vic[1, idx]:  # left deviation is more influential to this victim drone
                target_id = np.argsort(tmp_pg_ranks_tar2)[-nb_t:]  # thus, select 2 target drones with the highest score with left deviation
                f = [-1, -1]
            else:  # if the influence for right and left deviation is the same
                # sum the total score of top 2 target drones in each scenario
                target_id1 = np.argsort(tmp_pg_ranks_tar1)[-nb_t:]
                target_id2 = np.argsort(tmp_pg_ranks_tar2)[-nb_t:]
                # select the deviation direction where the total score is higher
                if np.sum(tmp_pg_ranks_tar1[target_id1]) > np.sum(tmp_pg_ranks_tar2[target_id2]):
                    target_id = target_id1
                    f = [1, 1]
                else:
                    target_id = target_id2
                    f = [-1, -1]

            victim_id = [idx, idx]  # for each victim drone, we have 2 target drones
            time = [start_t, start_t]
            pool[2 * i:2 * i + 2, :] = np.column_stack([target_id, victim_id, f, time])  # seedpool
        # Delete any zero rows
        pool = pool[~np.all(pool == 0, axis=1)]
        np.savetxt(seedpool, pool, delimiter=',', fmt='%1.2f')
        print(pool)

    # If only one direction of deviation is valid
    elif len(valid_idx) == 1:
        flag = 1
        for i in range(nb_v):
            idx = vic_id_tmp[i]
            tmp_pg_ranks_tar = pgrank_tar[valid_idx, :].copy()
            tmp_pg_ranks_tar[idx] = -1
            target_id = np.argsort(tmp_pg_ranks_tar)[-nb_t:]
            if valid_idx == 0:
                f = [1, 1]
            else:
                f = [-1, -1]

            victim_id = [idx, idx]
            time = [start_t, start_t]
            pool[2 * i:2 * i + 2, :] = np.column_stack([target_id, victim_id, f, time])
        # Delete any zero rows
        pool = pool[~np.all(pool == 0, axis=1)]
        np.savetxt(seedpool, pool, delimiter=',', fmt='%1.2f')
        print(pool)

    else:
        flag = 0

    return flag


def build_graph(dev, goal, pos_att, nb):
    A = np.zeros((nb, nb))
    flags = 1

    for i in range(nb):
        for j in range(i + 1, nb):
            if (pos_att[0, i] < pos_att[0, j] and dev == 1) or (pos_att[0, i] > pos_att[0, j] and dev == -1):
                weight = compute_weight(pos_att[:, i], pos_att[:, j], goal[i])
                A[i, j] = weight
                A[j, i] = weight
            else:
                A[i, j] = -1
                A[j, i] = -1

    return A, flags


def compute_weight(pos_i, pos_j, goal_i):
    dist = np.linalg.norm(pos_i - pos_j)
    angle = np.arctan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
    weight = dist / (1 + np.abs(np.sin(angle - goal_i)))

    return weight


# Example usage:
# nb, start_t, pos_att, dist_obs, final_dire = 5, 0.1, np.random.rand(2, 5), np.random.rand(5), np.random.choice([-1, 1], 5)
# seedpool = "seedpool.csv"
# flag = seed_gen(nb, start_t, pos_att, dist_obs, final_dire, seedpool)
# print("Flag:", flag)
