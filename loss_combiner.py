import numpy as np
import pandas as pd
import sys

base_name = 'losses/loss_tracker_mnist_naive'

dfs = []

for i in range(20):
    curr_name = base_name + "_" + str(i) + ".txt"
    dfs.append(pd.read_csv(curr_name,names=['epoch', 'size', 'loss']))

def get_losses(df):
    epochs = list(set(df['epoch']))
    losses_ret = [0 for e in epochs]
    sizes_ret = [0 for e in epochs]
    stds_ret = [0 for e in epochs]
    for e in epochs:
        losses_ret[e] = df[df['epoch'] == e]['loss'].mean()
        stds_ret[e] = df[df['epoch'] == e]['loss'].std()
        sizes_ret[e] = df[df['epoch'] == e]['size'].mean()
    return (np.array(sizes_ret), np.array(losses_ret), np.array(stds_ret))

df = pd.concat(dfs)
sizes, losses, stds = get_losses(df)
f = open(base_name + ".txt", 'a')
f.write(str(sizes))
f.write("\n")
f.write(str(losses))
f.write("\n")
f.write(str(stds))
f.write("\n")
f.close()