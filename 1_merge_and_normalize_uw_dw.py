import os

import numpy as np
import pandas as pd
import glob
import tqdm

INPUT_FOLDER = "data"

def remove_unnecessary_cols_dw(df):
    df = df.iloc[1:]
    return df.drop(columns=['Index', 'SN', 'INTTIME(ED)', 'SAMPLE(DELAY)',
                            'DARK_SAMP(ED)', 'DARK_AVE(ED)', 'SPECTEMP','FRAME(COUNTER)', 'TIMER',
                            'CHECK(SUM)', 'DATETAG'])
def remove_unnecessary_cols_uw(df):
    df = df.iloc[1:]
    return df.drop(columns=['Index', 'SN', 'INTTIME(EU)', 'SAMPLE(DELAY)',
                            'DARK_SAMP(EU)', 'DARK_AVE(EU)', 'SPECTEMP','FRAME(COUNTER)', 'TIMER',
                            'CHECK(SUM)', 'DATETAG'])

def simplify_column_names(df):
    curve_columns = []
    for e in df.columns:
        if e != "TIMETAG2":
            part = e.split("(")[1]
            part2 = part.split(")")[0]
            part3 = part2.split(".")[0]
            curve_columns.append(part3)
        else:
            curve_columns.append("TIMETAG2")
    df.columns = curve_columns
    return df
'''
    Reading and merging DW and UW data for each experiment
'''
all_data_for_modeling = []
for folder in tqdm.tqdm(glob.glob("{}/*".format(INPUT_FOLDER))): # for every measurment folder
    if os.path.isfile("{}/spectra_merged.csv".format(folder)):
        apogee = pd.read_csv("{}/spectra_merged.csv".format(folder))
        apogee['TIMETAG2'] = pd.to_datetime(apogee['TIMETAG2'])
        # DW
        dw = pd.read_csv("{}/msm_dw.tsv".format(folder), skiprows=3, delimiter='\t')
        dw['TIMETAG2'] = pd.to_datetime(dw['TIMETAG2'])
        dw = remove_unnecessary_cols_dw(dw)
        dw = simplify_column_names(dw)
        _, location, date, time = folder.split("/")[1].split("_")


        # UW
        uw = pd.read_csv("{}/msm_uw.tsv".format(folder), skiprows=3, delimiter='\t')
        uw['TIMETAG2'] = pd.to_datetime(uw['TIMETAG2'])
        uw = remove_unnecessary_cols_uw(uw) # DW is more imporatan so we are taking their columns
        uw.columns = dw.columns
        _, location, date, time = folder.split("/")[1].split("_")
        # Normalize by apogee
        # Find common columns between all 3
        common_columns = dw.columns.intersection(apogee.columns)
        dw = dw[common_columns]
        uw = uw[common_columns]
        # DW
        merged = pd.merge_asof(dw, apogee.sort_values('TIMETAG2'), on='TIMETAG2',
                               tolerance=pd.Timedelta('3s'), suffixes=("_DW", "_apogee"),
                               direction='nearest', allow_exact_matches=True).dropna()

        dw_columns = [col for col in merged.columns if col.endswith('_DW')]
        apogee_columns = [col for col in merged.columns if col.endswith('_apogee')]

        normalized_data_dw = pd.DataFrame(np.array(merged[dw_columns].values, dtype='float32')/
                                       np.array(merged[apogee_columns].values, dtype="float32"))
        normalized_data_dw['TIMETAG2'] = merged['TIMETAG2']
        normalized_data_dw.columns = common_columns
        normalized_data_dw = normalized_data_dw.dropna()
        # UW
        merged = pd.merge_asof(uw, apogee.sort_values('TIMETAG2'), on='TIMETAG2',
                               tolerance=pd.Timedelta('3s'), suffixes=("_UW", "_apogee"),
                               direction='nearest', allow_exact_matches=True).dropna()

        uw_columns = [col for col in merged.columns if col.endswith('_UW')]
        apogee_columns = [col for col in merged.columns if col.endswith('_apogee')]

        normalized_data_uw = pd.DataFrame(np.array(merged[uw_columns].values, dtype='float32')/
                                          np.array(merged[apogee_columns].values, dtype="float32"))
        normalized_data_uw['TIMETAG2'] = merged['TIMETAG2']
        normalized_data_uw.columns = common_columns
        normalized_data_uw = normalized_data_uw.dropna()
        for depthfile in glob.glob("{}/*.asc".format(folder)):
            depth = pd.read_csv(depthfile, skiprows=11)
            depth.columns = ['Temp', 'Meter', 'Date', 'TIMETAG2']
            depth['TIMETAG2'] = pd.to_datetime(depth['TIMETAG2'].str.strip())

        normalized_data_uw = pd.merge_asof(normalized_data_uw, depth.sort_values('TIMETAG2'), on='TIMETAG2',
                               tolerance=pd.Timedelta('3s'),
                               direction='nearest', allow_exact_matches=True)
        normalized_data_dw = pd.merge_asof(normalized_data_dw, depth.sort_values('TIMETAG2'), on='TIMETAG2',
                                           tolerance=pd.Timedelta('3s'),
                                           direction='nearest', allow_exact_matches=True)

        normalized_data_dw['location'] = location
        normalized_data_dw['date'] = date
        normalized_data_dw['sensor'] = 'DW'
        normalized_data_uw['location'] = location
        normalized_data_uw['date'] = date
        normalized_data_uw['sensor'] = 'UW'

        total_data = pd.concat([normalized_data_uw, normalized_data_dw])
        total_data.to_csv("{}/everything_merged_and_normalized.csv".format(folder))
        all_data_for_modeling.append(total_data)
    else:
        print("Error: Merged Apogee data was supposed to be in the folder")

all_data_for_modeling_df = pd.concat(all_data_for_modeling)
all_data_for_modeling_df['Date'] = pd.to_datetime(all_data_for_modeling_df['Date'])
all_data_for_modeling_df['Month'] = all_data_for_modeling_df['Date'].dt.month
all_data_for_modeling_df = all_data_for_modeling_df[all_data_for_modeling_df['Meter'] > 1]
all_data_for_modeling_df.to_csv("ALL_DATA.csv")