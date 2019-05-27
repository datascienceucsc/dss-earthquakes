# feature-extraction.py
#
# Anders Poirel 
# 26-05-2019

import pandas as pd
import numpy as np
import gc as gc
from glob import glob
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from numpy.fft import rfft

def get_predictors(filepath, col_name, seg_len, data_len):
    num_features = 15
    num_segs = data_len // seg_len + 1
    predictors = np.empty((num_segs, num_features))
    i = 0
    
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name],
                                chunksize = seg_len, dtype = np.int16)):
        # calculate features from seg
        predictors[i, 0] = seg.max()
        predictors[i, 1] = seg.sum()
        predictors[i, 2] = seg.mean()
        predictors[i, 3] = seg.std()
        predictors[i, 4] = seg.kurtosis()
        predictors[i, 5] = seg.skew()
        predictors[i, 6] = seg.quantile(q = 0.10)
        predictors[i, 7] = seg.quantile(q = 0.5)
        predictors[i, 8] = seg.quantile(q = 0.75)
        predictors[i, 9] = seg.quantile(q = 0.90)
        predictors[i, 10] = seg.quantile(q = 0.99)
        predictors[i, 11] = seg.mad()
        predictors[i, 12] = seg.sem()
 
        transform = rfft(seg)
        predictors[i,13] = transform.mean()
        predictors[i,14] = transform.std()

        i += 1
    
    imputer = SimpleImputer(strategy = 'median') 
    return imputer.fit_transform(predictors)


def get_responses(filepath, col_name, seg_len, data_len):
    num_segs = data_len // seg_len + 1
    responses = np.empty(num_segs)
    i = 0
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name], 
                                chunksize = seg_len, dtype = np.float16)):
        responses[i] = seg.values[-1]
        i += 1

    return responses


def get_test_predictors(file_directory, seg_len, num_segs):
    num_features = 14
    test_predictors = np.empty((num_segs, num_features)) 
    i = 0

    for fname in tqdm(glob(file_directory)):
        seg  = pd.read_csv(fname, dtype = np.int16)
        # calculate features from seg
        test_predictors[i, 0] = seg.max()
        test_predictors[i, 1] = seg.sum()
        test_predictors[i, 2] = seg.mean()
        test_predictors[i, 3] = seg.std()
        test_predictors[i, 4] = seg.kurtosis()
        test_predictors[i, 5] = seg.skew()
        test_predictors[i, 6] = seg.quantile(q = 0.10)
        test_predictors[i, 7] = seg.quantile(q = 0.5)
        test_predictors[i, 8] = seg.quantile(q = 0.75)
        test_predictors[i, 9] = seg.quantile(q = 0.90)
        test_predictors[i, 10] = seg.quantile(q = 0.99)
        test_predictors[i, 11] = seg.mad()
        test_predictors[i, 12] = seg.sem()

        transform = rfft(seg)
        test_predictors[i, 13] = transform.mean()
        test_predictors[i, 14] = transform.median()
        i += 1


    imputer = SimpleImputer(strategy = 'median')
    return imputer.fit_transform(test_predictors)

SEG_LEN = 150000      # Length of a segment of test data
DATA_LEN = 629145480  # Length of the entire time series
NUM_SEGS = 2624       # number of data segments in raw/train

X_train = get_predictors('../../data/raw/train.csv', 'acoustic_data', SEG_LEN, DATA_LEN)
np.savetxt("../../data/processed/X_train.csv", X_train, delimiter = ',')

y_train = get_responses('../../data/raw/train.csv', 'time_to_failure', SEG_LEN, DATA_LEN)
np.savetxt("../../data/processed/y_train.csv", y_train, delimiter = ',')

X_test = get_test_predictors('../../data/raw/test/*.csv', SEG_LEN, NUM_SEGS)
np.savetxt("../../data/processed/X_test.csv", X_test, delimiter = ',')

