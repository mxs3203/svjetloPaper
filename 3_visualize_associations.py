import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme(style="whitegrid")
data = pd.read_csv("ALL_DATA_with_sigs_k=5.csv")
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
data = data.iloc[:, 151:] # I do not need ccurves
data['Date'] = pd.to_datetime(data['Date'])
data['TIMETAG2'] = pd.to_datetime(data['TIMETAG2'])
data['Meter'] = data['Meter'].astype('float')
data['Temp'] = data['Temp'].astype('float')

# Extract time from TIMETAG2
data['Time'] = data['TIMETAG2'].dt.time

# Combine Date and Time
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

ax = sb.pointplot(data, y='Meter', x='Temp',hue='Month', palette='deep')
plt.xlabel('Sea column temperature')
ax.invert_yaxis()
plt.ylabel('Depth(m)')
plt.show()
#plt.savefig("plots/temperature_months.pdf", format="pdf")