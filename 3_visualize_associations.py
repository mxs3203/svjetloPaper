import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm

sb.set_theme(style="whitegrid")
data = pd.read_csv("ALL_DATA_with_sigs_k=5.csv")
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
data = data.iloc[:, 151:] # I do not need ccurves
# correct data types
data['Date'] = pd.to_datetime(data['Date'])
data['TIMETAG2'] = pd.to_datetime(data['TIMETAG2'])
data['Meter'] = data['Meter'].astype('float')
data['Temp'] = data['Temp'].astype('float')

# Fix time
data['Time'] = data['TIMETAG2'].dt.time
data['TIMETAG2'] = data.apply(lambda row: pd.Timestamp.combine(row['Date'].date(), row['Time']), axis=1)
data = data.drop(columns=['Time', 'Date'])
data['Month'] = pd.Categorical(data['Month'])

'''
 Basic plots, characterize two locations
'''
# sb.boxplot(data, x='Month', y='Temp',hue='Month', palette='deep')
# plt.xlabel('Month')
# plt.ylabel('Sea column temperature')
# plt.savefig("plots/temperature_months.pdf", format="pdf")

# F3, signatures per month and sensor
# tmp = data.melt(id_vars=['Month', 'location', 'sensor'], value_vars=['S1', 'S2', 'S3', 'S4', 'S5'])
# fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # Adjust figsize as needed
# sb.barplot(tmp[tmp['sensor'] == 'UW'], y='value', x='Month',hue='variable', palette='deep', ax=axes[0])
# plt.xlabel("UW Sensor")
# plt.ylabel('Signature Enrichment')
# sb.barplot(tmp[tmp['sensor'] == 'DW'], y='value', x='Month',hue='variable', palette='deep', ax=axes[1])
# plt.xlabel("DW Sensor")
# plt.ylabel('Signature Enrichment')
# plt.savefig("plots/F3_signatures_per_months_and_sensor.pdf", format="pdf")

# F3, binning the Depth
bin_edges = range(0, int(data['Meter'].max()) + 11, 10)
data['Meter_Bins'] = pd.cut(data['Meter'], bins=bin_edges, right=False)
#
# fig, axes = plt.subplots(2, 5, figsize=(19, 10))
# axes = axes.flatten()
# ax_cnt = 0
# for s in ['DW', 'UW']:
#     for m in [5,6,7,8,12]:
#         summary_df = data[(data['sensor'] == s) & (data['Month'] == m)].groupby(['Meter_Bins']).agg({
#             'S1': 'median',
#             'S2': 'median',
#             'S3': 'median',
#             'S4': 'median',
#             'S5': 'median'
#         }).reset_index()
#         print()
#         tmp = summary_df.melt(id_vars=['Meter_Bins'], value_vars=['S1', 'S2', 'S3', 'S4', 'S5'])
#         sb.barplot(tmp, x='value', y='Meter_Bins', hue='variable',palette='deep', ax=axes[ax_cnt])
#         #axes[ax_cnt].invert_yaxis()
#         axes[ax_cnt].set_title(f"Sensor: {s}, Month: {m}")
#         ax_cnt = ax_cnt + 1
# plt.tight_layout()
# plt.savefig("plots/F3_signatures_summarized_median_per_meter_bin.pdf", format="pdf")
#plt.show()

#
# fig, axes = plt.subplots(2, 5, figsize=(19, 8))
# axes = axes.flatten()
# ax_cnt = 0
# for s in ['DW','UW']:
#     for m in tqdm([5,6,7,8,12]):
#         summary_df = data[(data['sensor'] == s) & (data['Month'] == m)]
#         tmp = summary_df.melt(id_vars=['Meter'], value_vars=['S1', 'S2', 'S3', 'S4', 'S5'])
#         sb.kdeplot(tmp, x='value', y='Meter', hue='variable',palette='deep', ax=axes[ax_cnt])
#         #axes[ax_cnt].set_xticklabels(axes[ax_cnt].get_xticklabels(), rotation=90)
#         axes[ax_cnt].set_title(f"Sensor: {s}, Month: {m}")
#         axes[ax_cnt].invert_yaxis()
#         ax_cnt = ax_cnt + 1
# plt.tight_layout()
# plt.savefig("plots/F3_signatures_kde.pdf", format="pdf")
#plt.show()

'''
    Summary by depth and months
'''
line_styles = {5:(0, ()), 6:(0, (5, 1)), 7:'-.', 8:':',12:' '}
marker_shapes = ['o', 's', 'd', 'x', '*']
for s in ['DW', 'UW']:
    plt.figure(figsize=(12, 8))
    #for m in [5,6,7,8,12]:
    summary_df = data[(data['sensor'] == s)]
 #   tmp = summary_df[['Month', 'Meter_Bins','S1','S2', 'S3','S4','S5']].groupby(['Month', 'Meter_Bins']).mean()
    tmp = summary_df.melt(id_vars=['Month', 'Meter_Bins'], value_vars=['S1', 'S2', 'S3', 'S4', 'S5'])
    for i, (month, linestyle) in enumerate(line_styles.items()):
        print(marker_shapes[i], month)
        subset_tmp = tmp[tmp['Month'] == month]
        sb.pointplot(data=subset_tmp, x='value', y='Meter_Bins', hue='variable',
                    marker=marker_shapes[i],linewidth=0.5,markersize=10)

    plt.xlabel("Enrichment")
    plt.ylabel("Depth Groups(m)")
    plt.xticks(rotation=90)
    plt.title("{}".format(s))
    plt.savefig("plots/F3_{}_enrichment_depth_month.pdf".format(s), format="pdf")
'''
    Stasa's Signature plots
'''
# fig, axes = plt.subplots(2, 5, figsize=(15, 7))
# axes = axes.flatten()
# ax_cnt = 0
# for s in ['DW', 'UW']:
#     for sig in ['S1', 'S2', 'S3', 'S4', 'S5']:
#         summary_df = data[(data['sensor'] == s)]
#         #summary_df[sig] = np.log(summary_df[sig]+1)
#         sb.lineplot(summary_df, x=sig, y='Meter', hue='Month',palette='deep', ax=axes[ax_cnt])
#         #axes[ax_cnt].set_xticklabels(axes[ax_cnt].get_xticklabels(), rotation=90)
#         axes[ax_cnt].invert_yaxis()
#         axes[ax_cnt].set_title(f"Sensor: {s}, Signature: {sig}")
#         ax_cnt = ax_cnt + 1
# plt.tight_layout()
# plt.savefig("plots/F3_signature_line_plots_stasa_made.pdf", format="pdf")
#plt.show()
