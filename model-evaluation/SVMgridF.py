# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:20:55 2019

@author: melcb01
"""

import os
import pandas as pd
import numpy as np
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

path = 'c:\\users\\melcb01\\.spyder-py3\\patsat'
os.chdir(path)
from PatSatCleanOrdinal import Xs, yH4  # yH2, yH3, yH4, y14a, y14b, y05a, y05b  # Xc, Xv, Xs, Xn

# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xs, yH4, test_size = 0.3, random_state=42)

# Choose the type of Classifier
from sklearn.svm import SVC
clf = SVC()

# Set the accuracy scoring methods
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, confusion_matrix
# from sklearn.metrics import average_precision_score

scorers = {
        'kappa_scorer':make_scorer(cohen_kappa_score),
        'accuracy_score': make_scorer(accuracy_score)
        }

refit_score='kappa_scorer'

parameters = {
            'C': [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],
            'kernel':['linear'],
            'shrinking':[False],
            'verbose': [1],
            'decision_function_shape':['ovr'],
            'random_state': [2],
            'class_weight':[None]
            }

from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
grid_search = GridSearchCV(clf, parameters, scoring=scorers, refit=refit_score, cv = 5, verbose=2)
grid_search.fit(X_train, y_train)

# Make the predictions
y_pred = grid_search.predict(X_test)

# Set the clf to the best combination of parameters
est = grid_search.best_estimator_
bp = grid_search.best_params_
bs = grid_search.best_score_

print("best params: " + str(bp))
print("best scores: " + str(bs))

acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.4%}".format(acc))

# Confusion matrix on the test data
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix of SVM on test data:')
print(pd.DataFrame(cm))
# columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])              

recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
print(np.mean(recall))
print(np.mean(precision))

#y_tests = 0
#test_scores = []
#estimates = clf.predict_proba(X_test)
#y_tests+=estimates