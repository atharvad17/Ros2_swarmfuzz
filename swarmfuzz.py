import os
import shutil

def swarmfuzz(seedStart, seedEnd, dev, nb):
    c_obs = [50, 150]  # hard code the location of the obstacle
    max_ite = 15  # Maximum number of iterations
    dur_t = 0.5  # Start value

    # Add paths
    current_folder = os.getcwd()
    examples_path = os.path.join(current_folder, '../examples/examples_swarm/')
    root_f = os.path.join(current_folder, '../')

    for seed in range(seedStart, seedEnd + 1):
        # Temp files
        pos_csv = os.path.join(root_f, f'fuzz/tmp_files/attack_pos{seed}.csv')
        dist_csv = os.path.join(root_f, f'fuzz/tmp_files/dist_obs{seed}.csv')
        col_csv = os.path.join(root_f, f'fuzz/tmp_files/nb_col{seed}.csv')
        info_csv = os.path.join(root_f, f'fuzz/tmp_files/col_info{seed}.csv')
        parent_csv = os.path.join(root_f, f'fuzz/tmp_files/parent{seed}.csv')
        neigh_csv = os.path.join(root_f, f'fuzz/tmp_files/neighbor{seed}.csv')

        # Output files
        ite_csv = os.path.join(root_f, f'fuzz/search/iteration{seed}.csv')
        cond_csv = os.path.join(root_f, f'fuzz/search/condition{seed}.csv')
        dire_csv = os.path.join(root_f, f'fuzz/prepare/dire{seed}.csv')
        seedpool_csv = os.path.join(root_f, f'fuzz/seedpools/pool{seed}.csv')
        f_out = os.path.join(root_f, f'fuzz/attResults/att_results{seed}.csv')

        # Delete previous temp files
        for file_path in [pos_csv, dist_csv, col_csv, info_csv, parent_csv, neigh_csv]:
            if os.path.isfile(file_path):
                os.remove(file_path)

        # -------- 1. Preparation ---------
        print(f"********** No attack. Running seed: {seed} **********")
        example_vasarhelyi(0, 0, 0, 0, 0, seed, pos_csv, dist_csv, col_csv, info_csv)
        start_t, pos_att, dist_obs, final_dire = prepare(nb, c_obs, pos_csv, dist_csv, dire_csv, col_csv, seed)

        # ---------2. Generate seedpool ---------
        if start_t > -1:
            print("*********** Swarm vulnerability graph generator **********")
            flag = seed_gen(nb, start_t, pos_att, dist_obs, final_dire, seedpool_csv)
        else:
            print(f"No seedpool for seed {seed}")

        # ----------3. Gradient descent ----------
        if os.path.isfile(seedpool_csv):
            print("*********** Gradient descent ***********")
            search_grad(seedpool_csv, dev, max_ite, f_out, dur_t, seed, pos_csv, dist_csv, col_csv, info_csv,
                        parent_csv, neigh_csv, ite_csv, cond_csv)

        # Delete previous temp files
        for file_path in [pos_csv, dist_csv, col_csv, info_csv, parent_csv, neigh_csv]:
            if os.path.isfile(file_path):
                os.remove(file_path)

# Define the missing functions (example_vasarhelyi, prepare, seed_gen, search_grad) before using this code.
