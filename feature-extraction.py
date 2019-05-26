#%%
import pandas as pd
import numpy as np
import gc as gc
from glob import glob
from tqdm import tqdm
from sklearn.impute import SimpleImputer

#%%
def get_predictors(filepath, col_name, seg_len, data_len):
    num_features = 12
    num_segs = data_len // seg_len + 1
    predictors = np.empty((num_segs, num_features))
    i = 0
    
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name],
                                chunksize = seg_len, dtype = np.int16)):
        # calculate features from seg
        predictors[i, 1] = seg.sum()
        predictors[i, 2] = seg.mean()
        predictors[i, 3] = seg.var()
        predictors[i, 4] = seg.kurtosis()
        predictors[i, 5] = seg.skew()
        predictors[i, 6] = seg.quantile(q = 0.25)
        predictors[i, 7] = seg.quantile(q = 0.5)
        predictors[i, 8] = seg.quantile(q = 0.75)
        predictors[i, 9] = seg.quantile(q = 0.95)
        predictors[i, 10] = seg.mad()
        predictors[i, 11] = seg.sem()
        predictors[i, 0] = seg.max() 
        i += 1
    
    imputer = SimpleImputer(strategy = 'median') 
    return imputer.fit_transform(predictors)

#%%
def get_responses(filepath, col_name, seg_len, data_len):
    num_segs = data_len // seg_len + 1
    responses = np.empty(num_segs)
    i = 0
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name], 
                                chunksize = seg_len, dtype = np.float16)):
        responses[i] = seg.values[-1]
        i += 1

    return responses

#%%
def get_test_predictors(file_directory, seg_len, num_segs):
    num_features = 12
    test_predictors = np.empty((num_segs, num_features)) 
    i = 0

    for fname in tqdm(glob(file_directory)):
        seg  = pd.read_csv(fname, dtype = np.int16)
        # calculate features from seg
        test_predictors[i, 0] = seg.max()
        test_predictors[i, 1] = seg.sum()
        test_predictors[i, 2] = seg.mean()
        test_predictors[i, 3] = seg.var()
        test_predictors[i, 4] = seg.kurtosis()
        test_predictors[i, 5] = seg.skew()
        test_predictors[i, 6] = seg.quantile(q = 0.25)
        test_predictors[i, 7] = seg.quantile(q = 0.5)
        test_predictors[i, 8] = seg.quantile(q = 0.75)
        test_predictors[i, 9] = seg.quantile(q = 0.95)
        test_predictors[i, 10] = seg.mad()
        test_predictors[i, 11] = seg.sem()
        i += 1

    imputer = SimpleImputer(strategy = 'median')
    return imputer.fit_transform(test_predictors)

#%%
SEG_LEN = 150000      # Length of a segment of test data
DATA_LEN = 629145480  # Length of the entire time series
NUM_SEGS = 2624       # number of data segments in input/train

#%%
X_train = get_predictors('input/train.csv', 'acoustic_data', SEG_LEN, DATA_LEN)
#%%
y_train = get_responses('input/train.csv', 'time_to_failure', SEG_LEN, DATA_LEN)
#%%
X_test = get_test_predictors('input/test/*.csv', SEG_LEN, NUM_SEGS)

# saves predictors and responses for easier access in the future
#%%
np.savetxt("y_train.csv", y_train, delimiter = ',')
np.savetxt("X_train.csv", X_train, delimiter = ',')
np.savetxt("X_test.csv", X_test, delimiter = ',')
