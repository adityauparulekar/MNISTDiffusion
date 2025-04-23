import pandas as pd

def process_weights_df(df):
    df = df.set_index(['image_idx'])
    df['grad_norm'] /= df['grad_norm'].mean()
    # df['grad_norm'] = df['grad_norm'].clip(lower=0, upper=8)
    # df['grad_norm'] /= df['grad_norm'].mean()
    # df['grad_norm'] = 1/df['grad_norm']
    M = df['grad_norm'].max()
    return (df, M)

def load_weights(f_name):
    f = open(f_name)
    lines = f.readlines()
    a = []
    for l in lines:
        for x in eval(l):
            a.append(x)
    df = pd.DataFrame(a)
    return process_weights_df(df)

df, M = load_weights('mnist_grads.txt')
print(M)