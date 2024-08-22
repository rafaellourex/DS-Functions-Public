import numpy as np


def calculate_removed_obs(df_init, df_fin):
    init_len = len(df_init)
    fin_len = len(df_fin)
    pct_rem = np.round(1 - (fin_len / init_len), 3) * 100
    nr_rem = init_len - fin_len
    print(f"# of observations removed: {nr_rem}")
    print(f"% of observations removed: {pct_rem}%")
