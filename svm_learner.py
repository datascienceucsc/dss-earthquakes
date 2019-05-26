# svm_learner.py

#%%
import pandas as pd
from numpy import savetxt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#%%
X_train = pd.read_csv('X_train.csv', header = None)
y_train = pd.read_csv('y_train.csv', header = None)
X_test = pd.read_csv('X_test.csv', header = None)

#%% Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



#%% Parameter Search
regressor = SVR(cache_size = 1000)
param_grid = [{'C': [0.5, 0.75, 1, 5], 
               'kernel': ['poly'],
               'degree': [2,3,4,5],
               'gamma': ['auto', 0.1, 0.5]
              },
              {'C': [0.5, 0.75, 1, 5], 
               'kernel': ['rbf'],
               'gamma' : ['auto', 0.1, 0.5]
              }
             ]
#%%
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = param_grid,
                           scoring = make_scorer(mean_absolute_error),
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_train)

#%%
y_test = grid_search.predict(X_test)
savetxt('svm_pred.csv', y_test, delimiter = ',')

#%% Results
grid_search.best_params_
#%%
mean_absolute_error(y_train, y_pred)