import numpy as np
import pandas as pd
import tsfresh as tsf

# lengths of segments from which to extra

def get_predictors(filepath, col_name, seg_length, skip_amount = 100):


    series_df = pd.read_csv(filepath, 
                            usecols = col_name,
                            dtype = np.float32
                           )

    series_df = series_df.iloc[::skip_amount]
    num_points = len(series_df.index)
    interval_length = num_points / (seg_length / skip_amount)
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
                            dtype = np.float32
                           )
    num_points = 
    
    responses = 
    return None
    pass