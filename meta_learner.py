# meta_learner.py

#%%
import pandas as pd
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#%%
rf_pred = pd.read_csv('rf_pred.csv', header = None)
xgb_pred = pd.read_csv('xgb_pred.csv', header = None)
svm_pred = pd.read_csv('svm_pred.csv', header = None)
knn_pred = pd.read_csv('knn_pred.csv', header = None)
X_train = rf_pred
X_train.append(xgb_pred)
X_train.append(svm_pred)
X_train.append(knn_pred)

y_train = pd.read_csv('y_train', header = None)

meta_regressor = LinearRegression()
meta_regressor.fit(X_train, y_train)
y_pred = meta_regressor.predict(X_test)

#%% Build Kaggle submission
seg_names = []
for fname in glob('input/test/*.csv'):
    seg_names.append(fname[11:21]) # modify to 13:24 if on kaggle kernel

#%%
submission = pd.DataFrame({'seg_id': seg_names, 'time_to_failure': y_pred})
submission.to_csv('submission.csv', index = False)