# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:54:34 2019

@author: melcb01
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

path = 'c:\\users\\melcb01\\.spyder-py3\\patsat'
os.chdir(path)
# Import data; Keras only works on Numpy arrays
from PatSatCleanOrdinal import Xn, yH  
# yH2, yH2s, yH25, yH2n, yH3, yH3s, yH35, yH3n, yH4, yH4s, yH45, yH4n
# y14a, y14as, y14a5, y14an, y14b, y14bs, y14b5, y14bn
# y05a, y05as, y05a5, y05an, y05b, y05bs, y05b5, y05bn 
# Xc, Xv, Xs, Xn, Xn5

# Transform Target variables into categorical for Binary classification
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
bins1 = (-1, 9.0, 10.0)  # -1, 4.0, 7.0, 9.0, 10.0
group_names1 = ['Low', 'Perfect']  # Low, Mid, High, Perfect
howcat1 = pd.cut(yH, bins1, labels=group_names1)
y = le.fit(howcat1)
labels = le.transform(howcat1)
features = Xn

# Fix random seed for reproducibility
seed = 17
np.random.seed(seed)
N_FOLDS = 5
MAX_EVALS = 5

# Split into training and testing data
train_features, test_features, train_labels, y_test = train_test_split(
                        features, labels, train_size=0.8, random_state=seed)
# Create a training and testing dataset
train_set = lgb.Dataset(data = train_features, label = train_labels)
test_set = lgb.Dataset(data = test_features, label = y_test)

# Get default hyperparameters
model = lgb.LGBMClassifier()
default_params = model.get_params()

# Remove number estimators because we set to training obs length in cv call
del default_params['n_estimators']

# Cross validation with early stopping
cv_results = lgb.cv(default_params, train_set, num_boost_round = 6130, 
                    early_stopping_rounds = 100, metrics = 'auc', 
                    nfold = N_FOLDS, seed = seed)

print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.' 
      .format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('The optimal number of boosting rounds (estimators) was {}.'
      .format(len(cv_results['auc-mean'])))

# Optimal number of esimators found in cv
model.n_estimators = len(cv_results['auc-mean'])

# Train and make predictions with model; Calculate accuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
history = model.fit(train_features, train_labels)
y_pred = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(y_test, y_pred)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))
# Converting probabilities into 1 or 0 for accuracy score
for i in range(0,len(y_pred)): 
    if y_pred[i]>=.5:       # setting threshold to .5 
       y_pred[i]=1 
    else: 
       y_pred[i]=0 
accuracy_lgb = accuracy_score(y_test, y_pred) 
print('The baseline model scores {:.5f} Accuracy on the test set.'.format(accuracy_lgb))

def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 6130, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = seed)
    
    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    return [score, hyperparameters, iteration]
score, params, iteration = objective(default_params, 1)
print('The cross-validation ROC AUC was {:.5f}.'.format(score))

# Create a default model
model = lgb.LGBMModel()
model.get_params()

# Hyperparameter grid
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
    }

# Dataframe for random search
random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))

com = 1
for x in param_grid.values():
    com *= len(x)
print('There are {} combinations'.format(com))
print('This would take {:.0f} years to finish.'.format((100 * com) / (60 * 60 * 24 * 365)))

import random
random.seed(50)

# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']
# Example of random params
random_params

def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 

random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])

# Get the best parameters
random_search_params = random_results.loc[0, 'params']

# Create, train, test model
model = lgb.LGBMClassifier(**random_search_params, random_state = 42)
model.fit(train_features, train_labels)

y_pred2 = model.predict_proba(test_features)[:, 1]

print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(y_test, y_pred2)))
random_results['params']

# Converting probabilities into 1 or 0 for accuracy score
for i in range(0,len(y_pred2)): 
    if y_pred2[i]>=.5:       # setting threshold to .5 
       y_pred2[i]=1 
    else: 
       y_pred2[i]=0 
accuracy_lgb2 = accuracy_score(y_test, y_pred2) 
print('The baseline model scores {:.5f} Accuracy on the test set.'.format(accuracy_lgb2))
