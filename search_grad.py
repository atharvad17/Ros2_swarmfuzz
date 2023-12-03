import os
import numpy as np
import pandas as pd

def search_grad(pool_f, val, max_ite, f_out, dur, seed, pos_csv, dist_csv, col_csv, info_csv, parent_csv, neigh_csv, ite_csv, cond_csv):
    # ---------- 1. Read the initial seeds and iterate them one by one -------
    seedpool = np.loadtxt(pool_f, delimiter=',')
    rows, cols = seedpool.shape
    attack_start, duration, attack_id, victim_id, deviation, final_col = 0, 0, 0, 0, 0, 0
    out_mat = np.zeros((rows, 8))
    init_fitness = 0
    
    # Delete previous tmp files
    if os.path.isfile(ite_csv):
        os.remove(ite_csv)
    if os.path.isfile(cond_csv):
        os.remove(cond_csv)
    
    # delta x,y
    st_del, t_del = 0.01, 0.01
    lr = 0
    
    for row in range(rows):
        # Initial seeds
        t_init = dur
        att_id = int(seedpool[row, 0])
        vim_id = int(seedpool[row, 1])
        dev_y = val * seedpool[row, 2]
        st_init = seedpool[row, 3]
        ite = 0

        while ite < max_ite:
            # Parent point
            print(f"Seed: {seed} Parent: {st_init}, {t_init}")
            example_vasarhelyi(st_init, t_init, att_id, vim_id, dev_y, seed, pos_csv, dist_csv, col_csv, info_csv)
            cal_summary(col_csv, info_csv, parent_csv)
            record = np.loadtxt(parent_csv)
            nb_col = record[2]
            fitness = record[4]
            if ite == 0:
                init_fitness = fitness

            min_time = record[3]
            ite_mat = [fitness, lr, st_init, t_init]
            np.savetxt(ite_csv, [ite_mat], delimiter=',', fmt='%1.2f', comments='', header='', footer='', append=True)

            if nb_col == 1:
                print(f"Seed: {seed} At {record[3]} collides, start time = {record[0]}, duration = {record[1]}")
                attack_start = record[0]
                duration = record[1]
                attack_id = att_id
                victim_id = vim_id
                deviation = dev_y
                break

            # neighbor for differential derivatives
            # f(x+st_del, y)
            st = st_init + st_del
            t = t_init

            example_vasarhelyi(st, t, att_id, vim_id, dev_y, seed, pos_csv, dist_csv, col_csv, info_csv)
            cal_summary(col_csv, info_csv, neigh_csv)
            new_record = np.loadtxt(neigh_csv)
            new_nb_col = new_record[2]
            x_fitness1 = new_record[4]

            if new_nb_col == 1:
                attack_start = new_record[0]
                duration = new_record[1]
                attack_id = att_id
                victim_id = vim_id
                deviation = dev_y
                break

            # f(x, y+t_del)
            st = st_init
            t = t_init + t_del

            example_vasarhelyi(st, t, att_id, vim_id, dev_y, seed, pos_csv, dist_csv, col_csv, info_csv)
            cal_summary(col_csv, info_csv, neigh_csv)
            new_record = np.loadtxt(neigh_csv)
            new_nb_col = new_record[2]
            y_fitness1 = new_record[4]

            if new_nb_col == 1:
                attack_start = new_record[0]
                duration = new_record[1]
                attack_id = att_id
                victim_id = vim_id
                deviation = dev_y
                break

            dst = (x_fitness1 - fitness) / st_del
            dt = (y_fitness1 - fitness) / t_del
            if dst == 0:
                dst = 1e-06
            if dt == 0:
                dt = 1e-06

            st_tmp = st_init
            t_tmp = t_init

            # Calculate lr
            loss = fitness - 0.5
            k = max(abs(dst), abs(dt))

            lr = loss / (2 * k * k)

            st_init = st_tmp - lr * dst  # w/o momentum
            t_init = max(t_tmp - lr * dt, 0)  # w/o momentum

            # if the calculated time is too long, give up
            if (st_init + t_init) > p_sim.end_time or st_init < 0:
                break

            ite = ite + 1

            # calculate whether it's possible or not
            max_improve = -(min_time - st_tmp) * dt
            # record the condition, the derivation is only valid at the certain point
            cond_mat = [min_time, dt, max_improve, loss, dst, st_tmp, t_tmp]
            np.savetxt(cond_csv, [cond_mat], delimiter=',', fmt='%1.2f', comments='', header='', footer='', append=True)
            if ite > 5 and dt < 0 and max_improve < loss:
                break

        if nb_col == 1 or new_nb_col == 1:
            # Collision occurs and record the output
            final_col = 1
            if attack_start == 0:
                attack_start = st_tmp
            if duration == 0:
                duration = t_tmp
            attack_id = att_id
            victim_id = vim_id
            deviation = dev_y
            # delete any zero rows
            out_mat = out_mat[~np.all(out_mat == 0, axis=1)]
            out_mat[row, :] = [final_col, init_fitness, fitness, attack_start, duration, attack_id, victim_id, deviation]
            break

        if attack_start == 0:
            attack_start = st_tmp
        if duration == 0:
            duration = t_tmp
        attack_id = att_id
        victim_id = vim_id
        deviation = dev_y

        # delete any zero rows
        out_mat = out_mat[~np.all(out_mat == 0, axis=1)]
        out_mat[row, :] = [final_col, init_fitness, fitness, attack_start, duration, attack_id, victim_id, deviation]

    # --------- Write the outputs to the file -----------
    # out_mat = [final_col, init_fitness, fitness, attack_start, duration, x_fitness1, attack_id, victim_id, deviation]
    np.savetxt(f_out, out_mat, delimiter=',', fmt='%1.2f', comments='', header='', footer='')


# This function should be replaced with the actual implementation of the example_vasarhelyi and cal_summary functions
def example_vasarhelyi(st_init, t_init, att_id, vim_id, dev_y, seed, pos_csv, dist_csv, col_csv, info_csv):
    pass


# This function should be replaced with the actual implementation of the cal_summary function
def cal_summary(col_csv, info_csv, parent_csv):
    pass


# Example usage:
# search_grad("pool.csv", 0.1, 100, "out.csv", 0.2, 42, "pos.csv", "dist.csv", "col.csv", "info.csv", "parent.csv", "neigh.csv", "ite.csv", "cond.csv")
