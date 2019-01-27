# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:08:14 2019

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
from PatSatCleanOrdinal import Xn, yH4  # yH2, yH3, yH4, y14a, y14b, y05a, y05b  # Xc, Xv, Xs, Xn

# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xn, yH4, test_size = 0.3, random_state=42)

# Set the accuracy scoring method
from sklearn.metrics import cohen_kappa_score, make_scorer
kappa_scorer = make_scorer(cohen_kappa_score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = ['Cart',
          'RF',
          'MLP']

clfs = [DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier()]

params = {models[0] : {
        'max_depth': np.arange(1, 60, 10),
        'min_samples_leaf': np.arange(5,100,10),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'criterion': ['gini','entropy'],
        'min_samples_split' : np.arange(5,100,10),
        'random_state': [2,7,12,17,42],
        'class_weight': ['balanced', None]},
        models[1] : {
        'n_estimators': [10, 100, 200, 500, 1000],
        'max_depth': np.arange(1, 60, 10),
        'min_samples_leaf': np.arange(5,100,10),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'criterion': ['gini','entropy'],
        'min_samples_split' : np.arange(5,100,10),
        'random_state': [2,7,12,17,42],
        'class_weight': ['balanced', 'balanced_subsample', None]},
        models[2] : {
        'hidden_layer_sizes': np.arange(10, 100, 10),
        'activation': ['relu', 'identity', 'logistic'],
        'solver': ['adam', 'lbfgs'],
        'alpha': np.logspace(-5,2,7),
        'batch_size': [32, 50, 80, 130, 200],
        'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ],
        'random_state': [2,7,12,17,42]}
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