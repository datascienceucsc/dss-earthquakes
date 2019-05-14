import pandas as pd
import numpy as np
import gc 
from glob import glob
from tqdm import tqdm
from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
from sklearn import RandomForestRegressor
from sklearn.metrics import accuracy_score

def get_predictors(filepath, col_name, seg_len, data_len):

    num_segs = data_len // seg_len
    id_col = np.zeros(seg_len)
    predictors = np.empty((num_segs, len(MinimalFCParameters())))
    i = 0   

    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name], 
                           chunksize = seg_len, dtype = np.int16)):
        seg['id'] = id_col
        seg_features = extract_features(seg, column_id = 'id',
                                            column_value = 'colname',
                                            default_fc_parameters = 
                                                MinimalFCParameters())
        i += 1
        predictors[i, :] = seg_features.values[1:]

    return predictors


def get_responses(filepath, col_name, seg_len, num_segs):

    responses = np.empty(num_segs)
    i = 0
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name], 
                                chunksize = seg_len, dtype = np.float16)):
        responses[i] = seg.values[-1]

    return responses


def get_test_predictors(file_directory, col_name, seg_len, num_segs):

    test_predictors = np.empty((num_segs, len(MinimalFCParameters()))) 
    id_col = np.zeros(seg_len)
    i = 0
    
    for fname in glob(file_directory):
        seg_df = pd.read_csv(fname, dtype = np.int16)
        
        seg_df['id'] = id_col
        temp_predictors = extract_features(seg_df, column_value = col_name,
                                           column_id = 'id',
                                           default_fc_parameters = MinimalFCParameters)
        test_predictors[i,:] = temp_predictors.values[1:]
        i += 1

    return test_predictors  


def train_model(X_train, y_train, MAX_DEPTH):
    rf = RandomForestRegressor(max_depth = MAX_DEPTH, )
    rf.fit(X_train, y_train)

    return rf