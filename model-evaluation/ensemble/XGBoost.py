# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:11:51 2019

@author: melcb01
"""

import time
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import os
import numpy as np
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

path = 'c:\\users\\melcb01\\.spyder-py3\\patsat'
os.chdir(path)
# Import data; Keras only works on Numpy arrays
from PatSatCleanOrdinal import Xn, yH  
# yH2, yH2s, yH25, yH2n, yH3, yH3s, yH35, yH3n, yH4, yH4s, yH45, yH4n
# y14a, y14as, y14a5, y14an, y14b, y14bs, y14b5, y14bn
# y05a, y05as, y05a5, y05an, y05b, y05bs, y05b5, y05bn 
# Xc, Xv, Xs, Xn, Xn5

# Transform Target variables into categorical for classification
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
bins1 = (-1, 9.0, 10.0)  # -1, 4.0, 7.0, 9.0, 10.0
group_names1 = ['Low', 'Perfect']  # Low, Mid, High, Perfect
howcat1 = pd.cut(yH, bins1, labels=group_names1)
y = le.fit(howcat1)
y = le.transform(howcat1)
X = Xn

# Create training and test sets
from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test =  train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
X_train, X_valid, y_train, y_valid =  train_test_split(X_t, y_t, test_size = 0.25, train_size =0.75, random_state=42)

clf = xgb.XGBClassifier()

param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

fit_params = {'eval_metric': ['logloss', 'auc'],
              'early_stopping_rounds': 10,
              'eval_set': [(X_valid, y_valid)]}

from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, confusion_matrix
scorers = {
        'kappa_scorer':make_scorer(cohen_kappa_score),
        'accuracy_score': make_scorer(accuracy_score)
        }

refit_score='kappa_scorer'

ransearch = RandomizedSearchCV(clf, param_grid, n_iter=100,
                            n_jobs=1, verbose=1, cv=4,
                            fit_params=fit_params,
                            scoring=scorers, refit=refit_score, random_state=42)

print("Randomized search...")
search_time_start = time.time()
# Pass fit params here instead?
ransearch.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)
# Make the predictions
y_pred = ransearch.predict(X_test)
best_score = ransearch.best_score_
best_params = ransearch.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
    
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix of XGBoost on test data:')
print(pd.DataFrame(cm))

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.4%}".format(acc))
  
precision = np.diag(cm) / np.sum(cm, axis = 0)
mp = np.mean(precision)
print("Precision: {:.4%}".format(mp))
recall = np.diag(cm) / np.sum(cm, axis = 1) 
mr = np.mean(recall)
print("Recall: {:.4%}".format(mr))
  