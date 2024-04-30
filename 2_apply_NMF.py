import os

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import glob
import tqdm
from sklearn.utils import shuffle
from sklearn.decomposition import NMF

OPTIMAL_K_RUN = False
data = pd.read_csv("ALL_DATA.csv")
curve_columns = data.columns[1:152]
data[curve_columns] = data[curve_columns].applymap(lambda x: max(0, x))

if OPTIMAL_K_RUN:
    results = []
    for k in tqdm.tqdm(range(2, 11)):
        for i in range(50):
            nmf = NMF(n_components=k, max_iter=50, random_state=i)
            nmf_data = shuffle(data[curve_columns])
            nmf.fit(nmf_data)
            results.append(pd.DataFrame([{'K':k, 'iter': i, 'error': nmf.reconstruction_err_}]))
            # Print the reconstruction error
            #print(f"K={k}: Reconstruction Error = {nmf.reconstruction_err_}")

    results_df = pd.concat(results, ignore_index=True)
    sb.boxplot(data=results_df, x='K', y = 'error')
    plt.show()

K = 5
nmf = NMF(n_components=K,  init='nndsvdar',
                                random_state=7, solver='mu',
                                beta_loss='kullback-leibler', max_iter=1000)
nmf_data = data[curve_columns]
W = nmf.fit_transform(nmf_data)
H = nmf.components_
tmp = pd.DataFrame(H).T
tmp.columns = ['S1', 'S2', 'S3', 'S4', 'S5']
tmp['nm'] = curve_columns.values
tmp = tmp.melt(id_vars=['nm'])
plt.figure(figsize=(25,12))
p = sb.lineplot(tmp, y='value', x='nm',
               hue='variable',
               palette=sb.color_palette("tab10"))
sb.set(font_scale=2)
p.set_xlabel("nm",fontsize=30)
p.set_ylabel("Value",fontsize=30)
plt.xticks(rotation=90)
plt.title("Signatures Based on all Data")
plt.show()

data['S1'] = W[:, 0]
data['S2'] = W[:, 1]
data['S3'] = W[:, 2]
data['S4'] = W[:, 3]
data['S5'] = W[:, 4]
data.to_csv("ALL_DATA_with_sigs_k={}.csv".format(K))