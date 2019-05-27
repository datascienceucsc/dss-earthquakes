# meta_learner.py
#
# Anders Poirel 
# 25-05-2019

import pandas as pd
from glob import glob
from sklearn.linear_model import LinearRegression

rf_pred = pd.read_csv('../../data/interim/rf_pred.csv', header = None)
xgb_pred = pd.read_csv('../../data/interim/xgb_pred.csv', header = None)
svm_pred = pd.read_csv('../../data/interim/svm_pred.csv', header = None)
knn_pred = pd.read_csv('../../data/interim/knn_pred.csv', header = None)
X_train = rf_pred
X_train.append(xgb_pred)
X_train.append(svm_pred)
X_train.append(knn_pred)
y_train = pd.read_csv('../../data/processed/y_train.csv', header = None)

meta_regressor = LinearRegression()
meta_regressor.fit(X_train, y_train)

rf_test = pd.read_csv('../../data/interim/rf_test.csv', header = None)
xgb_test = pd.read_csv('../../data/interim/xgb_test.csv', header = None)
svm_test = pd.read_csv('../../data/interim/svm_test.csv', header = None)
knn_test = pd.read_csv('../../data/interim/knn_test.csv', header = None)
X_test = rf_test
X_test.append(xgb_test)
X_test.append(svm_test)
X_test.append(knn_pred)

y_pred = meta_regressor.predict(X_test)

# Build Kaggle submission
submission = pd.read_csv('../../models/submission.csv')

submission['time_to_failure'] = y_pred
submission.to_csv('../../models/submission.csv', index = False)