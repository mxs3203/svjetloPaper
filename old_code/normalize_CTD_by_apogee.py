import os

import pandas as pd
from tqdm import tqdm
import numpy as np

info = {
    1.1: 'Cast1',1.2: 'Cast1',
    '1.1': 'Cast1','1.2': 'Cast1',
    6: 'Cast6',  '6': 'Cast6',
    2.1: 'Cast2',2.2: 'Cast2',
    '2.1': 'Cast2','2.2': 'Cast2',
    3: 'Cast3','3': 'Cast3',
    7.0: 'Cast7',7: 'Cast7', '7': 'Cast7',
    8.0: 'Cast8',8: 'Cast8', '8': 'Cast8',
    'Jezero4': 'Jezero4',
    'Jezero5': 'Jezero5'
}


for folder in ['Cast1', 'Cast2','Cast3', 'Cast6', 'Cast7', 'Cast8', 'Jezero4', 'Jezero5']:
    if os.path.isfile("data/{}/spectra_merged.csv".format(folder)):
        print("File already exists... Skipping {}".format(folder))
    else:
        print("Merging {} apogee data...".format(folder))
        files_to_read = os.listdir("data/{}/SpectWDATA/SpectWDATA/".format(folder))
        spect_rows = []
        for f in files_to_read:
            apogee = pd.read_csv("data/{}/SpectWDATA/SpectWDATA/{}".format(folder, f), skiprows=1,sep=' ')
            apogee.columns = ['ignore1', 'nm', 'ignore2', 'value']
            apogee = apogee.drop(columns=['ignore1', 'ignore2'])
            apogee = apogee[apogee['nm'].between(350, 850)].T
            apogee = apogee[apogee.columns[::2]] # take every second column, so I have only round nms
            cols = apogee.head(1).values.tolist()[0]
            cols = [int(c) for c in cols]
            apogee = apogee.tail(1)
            time = f.split("#")[1].split(".")[0].replace("_", ":")
            apogee['TIMETAG2'] = time
            spect_rows.append(apogee)
        spect_data = pd.concat(spect_rows)
        cols.append("TIMETAG2")
        spect_data.columns = cols
        spect_data.to_csv("data/{}/spectra_merged.csv".format(folder))


data = pd.read_csv('data/forNMF_350_850.csv', low_memory=False)
data['TIMETAG2'] = pd.to_datetime(data['TIMETAG2'])
curve_columns = []
for e in data.columns[0:150]:
    part = e.split("(")[1]
    part2 = part.split(")")[0]
    part3 = part2.split(".")[0]
    curve_columns.append(part3)
[curve_columns.append(i) for i in data.columns[150:154]]
data.columns = curve_columns


# Filter data by something?
data = data[~data['Cast'].isin(['1.1', '1.2'])]

normalized_data = []
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    #depth data
    depth = pd.read_csv("data/{}/depth.asc".format(info[row['Cast']]), skiprows=11)
    depth.columns = ['Temp', 'Meter', 'Date', 'TIMETAG2']
    depth['TIMETAG2'] = pd.to_datetime(depth['TIMETAG2'].str.strip())
    # load appropriate spectra file
    apogee_data = pd.read_csv("data/{}/spectra_merged.csv".format(info[row['Cast']]))
    apogee_data['TIMETAG2'] = pd.to_datetime(apogee_data['TIMETAG2'])
    # find common nms between spectra and CTD
    common_columns = data.columns.intersection(apogee_data.columns)
    # filter both data frames by common nms and take current row od data
    apogee_data = apogee_data[common_columns]
    row = data.filter(items=[index], axis=0)
    tmp_row = row[common_columns]
    # merge those two dataframe by time with tolerance of 5s
    merged = pd.merge_asof(tmp_row, apogee_data.sort_values('TIMETAG2'), on='TIMETAG2',
                            tolerance=pd.Timedelta('2s'), suffixes=("_CTD", "_apogee"),
                           direction='nearest', allow_exact_matches=True)
    merged = pd.merge_asof(merged, depth.sort_values('TIMETAG2'), on='TIMETAG2',
                            tolerance=pd.Timedelta('2s'),
                           direction='nearest', allow_exact_matches=True)
    if merged.shape[0] != 0:
        # get columns that represnt ctd/apogee data
        ctd_columns = merged.columns[merged.columns.str.endswith('CTD')]
        apogee_columns = merged.columns[merged.columns.str.endswith('apogee')]
        # get two curves and normalize ctd by apogee
        normalized_curve = pd.DataFrame((merged[ctd_columns].values/(merged[apogee_columns].values+1)) *100)
        normalized_curve[normalized_curve < 0] = 0
        normalized_curve['TIMETAG2'] = merged['TIMETAG2']
        normalized_curve.columns = common_columns
        normalized_curve['Meter'] = merged['Meter']
        normalized_curve['Temp'] = merged['Temp']
        normalized_curve['Sensor'] = row['Sensor'].values
        normalized_curve['Cast'] = row['Cast'].values
        normalized_curve['Weather'] = row['Weather'].values
        normalized_data.append(normalized_curve)
    else:
        print("Not properly merged... lack of data...")
    #print(normalized_curve)


normalized_data_nmf = pd.concat(normalized_data)
print(normalized_data_nmf)
normalized_data_nmf.to_csv("data/forNMF_350_850_normalized_by_apogee.csv")

