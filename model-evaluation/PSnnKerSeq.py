# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:18:51 2019

@author: melcb01
"""

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


# Fix random seed for reproducibility
seed = 17
np.random.seed(seed)

# Import models, wrapper, validation, and validation evaluation
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Build Keras function that defines model, compiles, & returns it
# Leave create_model() empty if not GridSearching
def create_model(optimizer='rmsprop', init='glorot_uniform', activation='relu',
                 dense_layers = (90,60)):
	# create model
	model = Sequential()
    # 1st number = neurons in hidden layer, input_dim = number of features
	model.add(Dense(90, input_dim=60, activation='relu'))
	model.add(Dense(60, activation='relu'))
    # Use sigmoid output for binary classification
	model.add(Dense(1, activation='sigmoid'))
	# loss = optimization score function; change loss if multi-class target
    # Specify optimizer in compile if not GridSearching
	model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                  metrics=['accuracy'])
	return model

# Create model (specify epochs & batches if not GridSearching params)
model = KerasClassifier(build_fn=create_model, verbose=0)

# Grid search epochs, batch size, layers, activation, initializer, & optimizer
from sklearn.model_selection import GridSearchCV
dense_layers = [(90,60)]
activation = ['relu']
optimizers = ['nadam', 'adam', 'adamax']
init = ['glorot_normal', 'normal', 'he_normal']
epochs = [40, 50]
batches = [20, 25, 35]
param_grid = dict(dense_layers=dense_layers, activation=activation, optimizer=optimizers, 
                  epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X, y)
# Summarize GridSearch results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
## Evaluate using 10-fold cross validation (use if not GridSearching params)
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X, y, cv=kfold)
#print(results.mean())

# Plot loss and accuracy for the training and validation set   
import matplotlib.pyplot as plt
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    # As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Create a Full Multiclass Report
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
# Multiclass or binary report
# If binary (sigmoid output), set binary parameter to True
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def full_multiclass_report(model,
                           X,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=True):
    
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(X, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)


# Split training and test sets with the categorical y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed)

# Capture the best params from GridSearch
params = grid_result.best_params_

# Create the model with the best params found
model = create_model(dense_layers=params['dense_layers'],
                     activation=params['activation'],
                     init=params['init'],
                     optimizer=params['optimizer'])

# Then train it and display the results (X_train, y_train)
history = model.fit(X_train,
                    y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    verbose = 0)

model.summary()
plot_history(history)
full_multiclass_report(model,
                       X_test,
                       y_test,
                       classes = le.inverse_transform(np.arange(2)))




