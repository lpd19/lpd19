# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:32:23 2022

@author: ftb19213
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
path = r'C:\Users\ftb19213\Desktop\PhD\2022\Data\SEA'
file = r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\Model_B.csv'
validation= r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\External_data_validation.csv'
data = pd.read_csv(file)
validation = pd.read_csv(validation)

#%%

# -- DEfining variables
X = data[[' API', 'Excipient1', 'Excipient 2', 'Excipient 3', 'Excipient 4','x10', 'x50', 'x90', 'SMD', 'a10', 'a50', 'a90', 's10', 's50', 's90', 'BD (g/ml)']]
y = data[['Class']]

# -- Splitting the data if we use train/test so sample the data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#%% KNN classifier

from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(algorithm="brute", n_neighbors=11, metric="mahalanobis", metric_params={'VI': np.cov(X)})
kNN.fit(X,y.values.ravel())
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
scores = cross_val_score(kNN, X, y.values.ravel(), scoring="roc_auc", cv=cv)

print("kNN %0.3f performance with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% Support Vector Machines

from sklearn import svm

SVM = svm.SVC(kernel = "rbf", C=3.2)
SVM.fit(X, y.values.ravel())
cv_SVM = ShuffleSplit(n_splits=10, test_size=0.25, random_state=50)
scores_SVM = cross_val_score(SVM, X, y.values.ravel(), scoring="roc_auc", cv=cv_SVM)

print("SVM %0.3f performance with a standard deviation of %0.2f" % (scores_SVM.mean(), scores_SVM.std()))
#%% Random Forest

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=115, min_samples_split=5, random_state=50).fit(X, y.values.ravel())

cv_RF = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc", cv=cv_RF)

print("RF %0.3f performance with a standard deviation of %0.2f" % (scores_RF.mean(), scores_RF.std()))

#%% Neural Network -- MLP Classifier

from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='sgd', alpha=4e-6, activation = 'logistic',
                   hidden_layer_sizes=(100,), random_state=1, max_iter=100000)

MLP.fit(X,y.values.ravel())
cv_MLP = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
scores_MLP = cross_val_score(MLP, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_MLP)

print("NN %0.3f performance with a standard deviation of %0.2f" % (scores_MLP.mean(), scores_MLP.std()))


#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X,y.values.ravel())

cv_gnb = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_gnb)

print("NB %0.3f performance with a standard deviation of %0.2f" % (scores_gnb.mean(), scores_gnb.std()))
#%% Logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(penalty="l1",C=58, random_state=42, max_iter=1000, solver="liblinear").fit(X, y.values.ravel())

cv_LR = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores_LR = cross_val_score(LR, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_LR)

print("LR %0.3f performance with a standard deviation of %0.2f" % (scores_LR.mean(), scores_LR.std()))

#%% AdaBoost

from sklearn.ensemble import AdaBoostClassifier

AB = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=50, learning_rate=0.1, random_state=0).fit(X, y.values.ravel())

cv_AB = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_AB)

print("AB %0.3f performance with a standard deviation of %0.2f" % (scores_AB.mean(), scores_AB.std()))

#%% Confusion matrix for MLP Classifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10, shuffle=True)
for train_index, test_index in kf.split(data):

   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]

   MLP.fit(X_train, y_train.values.ravel())
   print(confusion_matrix(y_test, MLP.predict(X_test)))

