import os
import pandas as pd
import glob
import tqdm

INPUT_FOLDER = "data"

'''
    Merging Apogee data for each experiment
'''
for folder in tqdm.tqdm(glob.glob("{}/*".format(INPUT_FOLDER))): # for every measurment folder
    if os.path.isfile("{}/spectra_merged.csv".format(folder)):
        print("File already exists... Skipping {}".format(folder))
    else:
        print("Merging {} apogee data...".format(folder))
        files_to_read = os.listdir()
        spect_rows = []
        for f in glob.glob("{}".format("{}/Apogee/*#*".format(folder))): # for every apogee
            apogee = pd.read_csv("{}".format(f), skiprows=1,sep=' ')
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
        spect_data.to_csv("{}/spectra_merged.csv".format(folder))


