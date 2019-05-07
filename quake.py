import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tsfresh as tsf
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.ensemble import RandomForestRegressor


series_df = pd.read_csv('train.csv', 
                        usecols = ['acoustic_data', 'time_to_failure'])
series_df = series_df.iloc[::100]

num_points = series_df.count
num_intervals= 16
interval_length = num_points //  num_intervals

id_col = []
for i in range(num_points):
    id[i] = num_points // num_intervals

series_df['id'] = id_col

acoustic_features = tsf.extract_features( series_df, 
                        column_sort = 'time_to_failure', 
                        column_value =  'acoustic_data',
                        column_id = 'id',
                        default_fc_parameters = MinimalFCParameters())

feature_labels = []
for i in range(num_intervals):
    feature_labels[i] = series_df.iloc[ (1/2 + i) * interval_length, 2]

