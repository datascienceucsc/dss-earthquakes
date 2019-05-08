# ts_extraction.py
#
# Anders Poirel
# 06-05-2019
#
#
# usage: save this file into directory containing main script
# use `from kextraction import get_predictors, get_responses`


import numpy as np
import pandas as pd
import tsfresh as tsf
import glob

def get_predictors(filepath, col_name, seg_length, skip_amount = 100):
    series_df = pd.read_csv(filepath, 
                            usecols = col_name,
                            dtype = np.float32
                           )

    series_df = series_df.iloc[::skip_amount]
    num_points = len(series_df.index)
    interval_length = num_points // (seg_length // skip_amount)
    num_segs = num_points // interval_length 

    id_col = np.empty((num_points ,1))
    for i in range(num_segs):
        id_col[i] = i // interval_length

    series_df['id'] = id_col

    from tsfresh.feature_extraction import MinimalFCParameters

    predictors = tsf.extract_features(series_df,
                                      column_value = col_name,
                                      column_id = 'id',
                                      default_fc_parameters = MinimalFCParameters()
                                     )
    return predictors.values[:, 1:]



def get_responses(filepath, col_name, seg_length, skip_amount = 100):

    series_df = pd.read_csv(filepath, 
                            usecols = col_name,
                            dtype = np.float32)

    series_df = series_df.iloc[::skip_amount]
    num_points = len(series_df.index)
    interval_length = num_points // (seg_length // skip_amount)
    num_segs = num_points // interval_length

    response = np.empty(num_segs)
    for i in range(num_segs):
        response[i] = series_df.iloc[interval_length * (i+1), 0]
    return response

def get_test_predictors(file_directory, col_name, seg_length, 
                        skip_amount = 100):

    test_predictors = []
    
    for fname in glob.glob(file_directory):
        seg_df = pd.read_csv(fname)
        id_col = np.zeroes(seg_length)
        seg_df['id'] = id_col

        temp_predictors = tsf.extract_features(seg_df,
                                               column_value = col_name,
                                               column_id = 'id')
        test_predictors.append(temp_predictors.iloc[1:])
    return np.array(test_predictors)