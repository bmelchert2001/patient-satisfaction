# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 22:37:52 2019

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

# Set the accuracy scoring method
from sklearn.metrics import cohen_kappa_score, make_scorer
kappa_scorer = make_scorer(cohen_kappa_score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = ['Cart',
          'SVCL']

clfs = [DecisionTreeClassifier(),
        SVC()]

params = {models[0] : {
        'max_depth': np.arange(1, 60, 10),
        'min_samples_leaf': np.arange(5,100,10),
        'max_features': ['auto', None],
        'criterion': ['gini','entropy'],
        'min_samples_split' : np.arange(5,100,5),
        'class_weight': ['balanced', None]},
        models[1] : {
        'C' : np.logspace(-5,3,8),
        'kernel':['linear'],
        'class_weight':['balanced', None]}   
        }

y_tests = 0
test_scores = []

from sklearn.model_selection import RandomizedSearchCV
for name, estimator in zip(models,clfs):
    print('Defined %d models' % len(models))
    print(name)
    clf = RandomizedSearchCV(estimator, params[name], scoring= kappa_scorer, verbose=2, n_iter= 10, cv=10)
    clf.fit(X_train, y_train)

    print("best params: " + str(clf.best_params_))
    print("best scores: " + str(clf.best_score_))
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, clf.predict(X_test))
    print("Accuracy: {:.4%}".format(acc))
    
    test_scores.append((acc, clf.best_score_, clf.best_params_))

