#%%
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

#%%
def get_predictors(filepath, col_name, seg_len, data_len):
    num_features = 13
    num_segs = data_len // seg_len + 1
    predictors = np.empty((num_segs, num_features))
    i = 0
    
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name],
                                chunksize = seg_len, dtype = np.int16)):
        # calculate features from seg
        predictors[i, 0] = seg.max()
        predictors[i, 1] = seg.min()
        predictors[i, 2] = seg.sum()
        predictors[i, 3] = seg.mean()
        predictors[i, 4] = seg.var()
        predictors[i, 5] = seg.kurtosis()
        predictors[i, 6] = seg.skew()
        predictors[i, 7] = seg.quantile(q = 0.25)
        predictors[i, 8] = seg.quantile(q = 0.5)
        predictors[i, 9] = seg.quantile(q = 0.75)
        predictors[i, 10] = seg.quantile(q = 0.95)
        predictors[i, 11] = seg.mad()
        predictors[i, 12] = seg.sem()
    
        i += 1

    return predictors

#%%
def get_responses(filepath, col_name, seg_len, num_segs):
    responses = np.empty(num_segs)
    i = 0
    for seg in tqdm(pd.read_csv(filepath, usecols = [col_name], 
                                chunksize = seg_len, dtype = np.float16)):
        responses[i] = seg.values[-1]

    return responses

#%%
def get_test_predictors(file_directory, col_name, seg_len, num_segs):
    num_features = 1000
    test_predictors = np.empty((num_segs, num_features)) 

    for fname in glob(file_directory):
        seg_df = pd.read_csv(fname, dtype = np.int16)
        # FIXME: interior similar to processing for a single chunk in get_predictors
        pass

    return test_predictors  

#%%
SEG_LEN = 150000      # Length of a segment of test data
DATA_LEN = 621985673  # Length of the entire time series

# FIXME: number of segments allocated in get_predictors seems insufficient,
#  get array OOB error at index 4147
X_train = get_predictors('input/train.csv', 'acoustic_data', SEG_LEN, DATA_LEN)

#%%
y_train = get_responses('input/train.csv', 'time_to-failure', SEG_LEN, DATA_LEN)

# saves predictors and responses for easier access in the future
#%%
np.savetxt("y_train.csv", y_train, delimiter = ",")
np.savetxt("X_train.csv", X_train, delimiter = ",")

#%%
rf = RandomForestRegressor(max_depth = 7, max_features = 'sqrt')
rf.fit(X_train, y_train)

#%%
y_fake_pred = rf.predict(X_train)
accuracy_score(y_train, y_pred)

#%%
X_test_train = get_test_predictors( SEG_LEN, DATA_LEN)
y_pred = rf.predict()

# Build submission CSV file using results
#%%
seg_names = []
for fnames in glob('input/test/*.csv'):
    seg_names.append(fname)

submission = pd.DataFrame({'seg_id': seg_names, 'time_to_failure': y_pred})
submission.to_csv('submission.csv', index = false)