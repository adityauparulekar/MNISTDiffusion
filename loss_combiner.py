import pandas as pd
import sys

base_name = 'losses/loss_tracker_mnist_naive_'

dfs = []

for i in range(20):
    curr_name = base_name + str(i) + ".txt"
    dfs.append(pd.read_csv(curr_name))


