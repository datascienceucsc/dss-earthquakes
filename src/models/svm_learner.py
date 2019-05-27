# svm_learner.py
# Anders Poirel
# 26-05-2019

import pandas as pd
from numpy import savetxt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X_train = pd.read_csv('../../data/processed/X_train.csv', header = None)
y_train = pd.read_csv('../../data/processed/y_train.csv', header = None)
X_test = pd.read_csv('../../data/processed/X_test.csv', header = None)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

regressor = SVR(cache_size = 1000)
param_grid = [{'C': [0.75, 1], 
               'kernel': ['poly'],
               'degree': [2,3,4],
               'gamma': ['auto', 0.1]
              },
              {'C': [0.75, 1], 
               'kernel': ['rbf'],
               'gamma' : ['auto', 0.1]
              }
             ]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = param_grid,
                           scoring = make_scorer(mean_absolute_error),
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_train)
savetxt('../../data/interim/svm_pred.csv', y_pred, delimiter = ',')

y_test = grid_search.predict(X_test)
savetxt('../../data/interim/svm_test.csv', y_test, delimiter = ',')

grid_search.best_params_
mean_absolute_error(y_train, y_pred)