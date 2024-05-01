
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr

data = pd.read_csv("ALL_DATA.csv")
curve_columns = data.columns[1:152]
sigs = pd.read_csv("contrastive_signatures.csv")
sigs.columns = ['index', 'S1', 'S2', 'S3', 'S4', 'S5','S6', 'S7', 'Month']
sig_columns = sigs.columns[1:8]
correlation_results = {}
cols = []
for signature_column in sigs[sig_columns].columns:
    correlations = []
    for curve_column in data[curve_columns].columns:
        correlation, p = pearsonr(data[curve_column][:len(sigs)], sigs[signature_column])
        correlations.append(correlation)

    correlation_results[signature_column] = correlations

# Convert correlation results to a DataFrame
correlation_df = pd.DataFrame(correlation_results, index=curve_columns)
correlation_df.to_csv("corrs.csv")

plt.figure(figsize=(10, 6))
sb.lineplot(y=correlation_df['S1'],x=correlation_df.index, label="S1")
sb.lineplot(y=correlation_df['S2'],x=correlation_df.index, label="S2")
sb.lineplot(y=correlation_df['S3'],x=correlation_df.index, label="S3")
sb.lineplot(y=correlation_df['S4'],x=correlation_df.index, label="S4")
sb.lineplot(y=correlation_df['S5'],x=correlation_df.index, label="S5")
sb.lineplot(y=correlation_df['S6'],x=correlation_df.index, label="S6")
sb.lineplot(y=correlation_df['S7'],x=correlation_df.index, label="S7")
plt.title('Heatmap of Nanometers vs Signatures')
plt.xlabel('Signatures')
plt.xticks(rotation=90, size =7)
plt.ylabel('Nanometers')
plt.show()