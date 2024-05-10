import os

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import glob
import tqdm
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.decomposition import NMF

OPTIMAL_K_RUN = False
data = pd.read_csv("ALL_DATA.csv")
curve_columns = data.columns[1:152]
scaler = MinMaxScaler()
data[curve_columns] = scaler.fit_transform(data[curve_columns])

if OPTIMAL_K_RUN:
    results = []
    for k in tqdm.tqdm(range(2, 11)):
        for init in ['random', 'nndsvd', 'nndsvda', 'nndsvdar']:
            for i in range(50):
                nmf = NMF(n_components=k, max_iter=50, random_state=i, init=init)
                nmf_data = shuffle(data[curve_columns])
                nmf.fit(nmf_data)
                results.append(pd.DataFrame([{'K':k, 'iter': i, 'error': nmf.reconstruction_err_, 'init':init}]))
                # Print the reconstruction error
            #print(f"K={k}: Reconstruction Error = {nmf.reconstruction_err_}")

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df[results_df['init'] == 'random']
    sb.boxplot(data=results_df, x='K', y='error')
    unique_K_values = sorted(results_df['K'].unique())
    for i in range(len(unique_K_values) - 1):
        group1 = results_df[results_df['K'] == unique_K_values[i]]['error']
        group2 = results_df[results_df['K'] == unique_K_values[i + 1]]['error']
        t_stat, p_value = ttest_ind(group1, group2)
        print(f"T-test between K={unique_K_values[i]} and K={unique_K_values[i + 1]}:")
        print(f"  t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
        plt.text(i + 0.5, max(group1.max(), group2.max()), round(p_value, 4),
                 ha='center',
                 va='bottom',
                 color='black')
    plt.savefig("plots/F2_optimal_k.pdf", format="pdf")
    #plt.show()

K = 5
nmf = NMF(n_components=K,  init='nndsvdar',
                                random_state=7, solver='mu',
                                beta_loss='frobenius', max_iter=1000)
nmf_data = data[curve_columns]
W = nmf.fit_transform(nmf_data)
H = nmf.components_
tmp = pd.DataFrame(H).T
tmp.columns = ['S1', 'S2', 'S3', 'S4', 'S5']
tmp['nm'] = curve_columns.values
tmp = tmp.melt(id_vars=['nm'])
tmp['nm'] = tmp['nm'].astype('int')
plt.figure(figsize=(10,7))
p = sb.lineplot(tmp, y='value', x='nm',
               hue='variable',
               palette=sb.color_palette("tab10"))
sb.set(font_scale=2)
p.set_xlabel("nm",fontsize=30)
p.set_ylabel("Value",fontsize=30)
plt.xticks(rotation=90)
plt.title("Signatures Based on all Data")
plt.savefig("plots/F2_signatures.pdf", format="pdf")

data['S1'] = W[:, 0]
data['S2'] = W[:, 1]
data['S3'] = W[:, 2]
data['S4'] = W[:, 3]
data['S5'] = W[:, 4]
data.to_csv("ALL_DATA_with_sigs_k={}.csv".format(K))