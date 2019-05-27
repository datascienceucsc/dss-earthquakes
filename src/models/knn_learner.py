# knn_learner.py
#
# Anders Poirel
# 26-05-2019

import pandas as pd
from numpy import savetxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

X_train = pd.read_csv('../../data/processed/X_train.csv', header = None)
y_train = pd.read_csv('../../data/processed/y_train.csv', header = None)
X_test = pd.read_csv('../../data/processed/X_test.csv', header = None)

knn_regressor = KNeighborsRegressor(algorithm = 'auto')
param_grid = {'n_neighbors' : [5, 10, 20],
              'leaf_size': [10, 30, 50]
             }

grid_search = GridSearchCV(estimator = knn_regressor,
                           param_grid = param_grid,
                           scoring = make_scorer(mean_absolute_error),
                           cv = 10,
                           n_jobs = -1
                           )
                      
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_train)
savetxt('../../data/interim/knn_pred.csv', y_pred, delimiter = ',')

y_test = grid_search.predict(X_test)
savetxt('../../data/interim/knn_test.csv', y_test, delimiter = ',')