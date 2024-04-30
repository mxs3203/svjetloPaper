import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

data = pd.read_csv("ALL_DATA.csv")
data['date'] = data['date'].astype(str)
curve_columns = data.columns[1:152]

window_size = 20

sequences = []
labels = []
for location in data.location.unique():
    for date in data.date.unique():
        for sensor in data.sensor.unique():
            print(location, date, sensor)
            tmp = data[(data['location'] == location) & (data['date'] == date) & (data['sensor'] == sensor)].sort_values('Meter')
            if np.shape(tmp)[0] > window_size + 1:
                for i in tqdm(range(len(tmp) - window_size + 1)):
                    sequence = tmp.iloc[i:i+window_size, 1:152].values
                    label = int(tmp.iloc[i+window_size-1, -1]) # Month is the last one
                    sequences.append(sequence)
                    labels.append(label)
sequences,labels = np.array(sequences), np.array(labels)

# Print the shape of sequences and labels
print("Shape of sequences:", sequences.shape)
print("Shape of labels:", labels.shape)
file_path = 'example_object.pkl'

with open("X.pck", 'wb') as file:
    pickle.dump(sequences, file)
with open("Y.pck", 'wb') as file:
    pickle.dump(labels, file)