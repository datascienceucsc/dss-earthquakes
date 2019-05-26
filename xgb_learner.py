# xgb_learner.py 

#%%
import pandas as pd
from numpy import savetxt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor

#%%
X_train = pd.read_csv('X_train.csv', header = None)
y_train = pd.read_csv('y_train.csv', header = None)
X_test = pd.read_csv('X_test.csv', header = None)

# I ran several iterations of grid search to tune the parameters, these are just the 
# parameters of the final grid search

#%% Parameter Grid Search
xgb_regressor = XGBRegressor()
param_grid = {'n_estimators' : [21,22,23,24,25],
              'max_depth'    : [1,2,3,4,5],
              'learning_rate': [0.1]
             }
#%%
grid_search = GridSearchCV(estimator = xgb_regressor,
                           param_grid = param_grid,
                           scoring = make_scorer(mean_absolute_error, 
                                                 greater_is_better= False),
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_train)

#%% 
y_test = grid_search.predict(X_test)
savetxt('xgb_pred.csv', y_test, delimiter = ',')

#%% Diagnostic
grid_search.best_params_
#%%
mean_absolute_error(y_train, y_pred)

