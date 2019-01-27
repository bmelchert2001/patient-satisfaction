# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:49:56 2019

@author: melcb01
"""

import os
import numpy as np
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Set working directory to file source path  # Read in Pickled DataFrame and variables
path = 'c:\\users\\melcb01\\.spyder-py3\\patsat'
os.chdir(path)
from PatSatCleanOrdinal import Xn, yH2  # yH2, yH3, yH4, y14a, y14b, y05a, y05b  # Xc, Xv, Xs, Xn

Y = yH2
X = Xn

# Multinomial classification spot check model
import warnings
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score, make_scorer

kappa_scorer = make_scorer(cohen_kappa_score)

# Create dictionary of standard models to evaluate {name:object}
def define_models(models=dict()):
    # Linear models
    penalty = ['l1', 'l2']
    clog = np.logspace(-3,1,4)
    for p in penalty:
        for c in clog:
            models['logistic- C='+str(c)+', Penalty='+str(p)] = LogisticRegression(solver= 'lbfgs', C=c, penalty=p, multi_class='multinomial')
    alpha = np.logspace(-4,1,3)
    for a in alpha:
        models['ridge-'+str(a)] = RidgeClassifier(alpha=a)
    alpha1 = np.logspace(-4,2,4)
    penalty1 = ["none", "l1", "l2"]
    for a in alpha1:
        for p in penalty1:
            models['sgd- Alpha='+str(a)+', Penalty='+str(p)] = SGDClassifier(alpha=a, penalty=p, max_iter=1000, tol=1e-3)
    models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    # Non-linear models
    n_neighbors = range(1, 21, 9)
    for k in n_neighbors:
        models['knn-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
    max_depth = [3, 5, 7]
    min_samples_leaf = [0.04, 0.08]
    max_features = [0.2, 0.5, 0.8]
    for d in max_depth:
        for s in min_samples_leaf:
            for f in max_features:
                models['cart- Depth='+str(d)+', Samples='+str(s)+', Features='+str(f)] = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s, max_features=f)
                models['extra- Depth='+str(d)+', Samples='+str(s)+', Features='+str(f)] = ExtraTreesClassifier(n_estimators=100, max_depth=d, min_samples_leaf=s, max_features=f)
                models['rf- Depth='+str(d)+', Samples='+str(s)+', Features='+str(f)] = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=s, max_features=f)
                models['gbm- Depth='+str(d)+', Samples='+str(s)+', Features='+str(f)] = GradientBoostingClassifier(n_estimators=100, max_depth=d, min_samples_leaf=s, max_features=f)
    cs = np.logspace(-3,1,3)
    gammas = [0.001, 0.05, 1]
    for c in cs:
        for g in gammas:
            models['svmr- C='+str(c)+', Gam='+str(g)] = SVC(C=c, gamma=g)
            models['svmp- C='+str(c)+', Gam='+str(g)] = SVC(kernel='poly', C=c, gamma=g)
        models['svml'+str(c)] = SVC(kernel='linear', C=c)
    nu = [.25, .5, .75]
    for n in nu:
        models['svmnu'+str(n)] = NuSVC(kernel='linear', nu=n)
    models['bayes'] = GaussianNB()
    # Ensemble models
    n_trees = 100
    models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
    models['bag'] = BaggingClassifier(n_estimators=n_trees)
    # Neural network
    a_values = np.logspace(-4,0,3)
    layer = [40, 60, (80,40)]
    for a in a_values:
        for l in layer:
            models['mlp- Alpha='+str(a)+', Layers='+str(l)] = MLPClassifier(hidden_layer_sizes=l, alpha=a)
    # Discriminant analysis
    models['lda'] = LinearDiscriminantAnalysis()
    models['qda'] = QuadraticDiscriminantAnalysis()
    print('Defined %d models' % len(models))
    return models

# Create feature preparation pipeline for model (try standard+normal, or just standard)
def make_pipeline(model):
	steps = list()
#	# standardization
#	steps.append(('standardize', StandardScaler()))
#	# normalization
#	steps.append(('normalize', MinMaxScaler()))
#	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# Evaluate only a single model
def evaluate_model(X, Y, model, folds, repeats, metric):
	# create the pipeline
	pipeline = make_pipeline(model)
	# evaluate model
	scores = list()
	# repeat model evaluation n times
	for _ in range(repeats):
		# perform run
		scores_r = cross_val_score(pipeline, X, Y, scoring=kappa_scorer, cv=folds, n_jobs=-1)
		# add scores to list
		scores += scores_r.tolist()
	return scores

# Evaluate model and try to trap errors and hide warnings
def robust_evaluate_model(X, Y, model, folds, repeats, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, Y, model, folds, repeats, metric)
	except:
		scores = None
	return scores

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, Y, models, folds=10, repeats=3, metric=kappa_scorer):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		scores = robust_evaluate_model(X, Y, model, folds, repeats, metric)
		# show process
		if scores is not None:
			# store a result
			results[name] = scores
			mean_score, std_score = mean(scores), std(scores)
			print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
		else:
			print('>%s: error' % name)
	return results

# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	# retrieve the top n for summarization
	names = [x[0] for x in mean_scores[:n]]
	scores = [results[x[0]] for x in mean_scores[:n]]
	# print the top n
	print()
	for i in range(n):
		name = names[i]
		mean_score, std_score = mean(results[name]), std(results[name])
		print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
	# boxplot for the top n
	pyplot.boxplot(scores, labels=names)
	_, labels = pyplot.xticks()
	pyplot.setp(labels, rotation=90)
	pyplot.savefig('spotcheck.png')


# get model list
models = define_models()
# evaluate models
results = evaluate_models(X, Y, models)
# summarize results
summarize_results(results)
