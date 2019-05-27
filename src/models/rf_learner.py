# rf_learner.py
#
# Anders Poirel
# 26-05-2019

import pandas as pd
from numpy import savetxt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv('../../data/processed/X_train.csv', header = None)
y_train = pd.read_csv('../../data/processed/y_train.csv', header = None)
X_test = pd.read_csv('../../data/processed/X_test.csv', header = None)
Parameter Grid Search
rf_regressor = RandomForestRegressor()
param_grid = {'n_estimators': [50,75,100,125],
              'max_depth'   : [3,4,5,6,7,],
              'max_features': ['auto', 'sqrt']
             }
grid_search = GridSearchCV(estimator = rf_regressor,
                           param_grid = param_grid,
                           scoring = make_scorer(mean_absolute_error, 
                                                 greater_is_better= False),
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_train)
savetxt('../../data/interim/rf_pred.csv', y_pred, delimiter = ', ')

y_test = grid_search.predict(X_test)
savetxt('../../data/interim/rf_test.csv', y_test, delimiter = ',')

grid_search.best_params_
mean_absolute_error(y_train, y_pred)