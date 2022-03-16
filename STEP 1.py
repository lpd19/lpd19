# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 08:31:06 2022

@author: ftb19213
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plts


#%%

data = pd.read_csv(file)
validation = pd.read_csv(validation)

#%%

# -- DEfining variables
X = data[['API', 'Excipient1', 'Excipient2', 'Excipient3', 'Excipient4','x10', 'x50', 'x90', 'SMD', 'a10', 'a50', 'a90', 's10', 's50', 's90', 'BD(g/ml)']]
y = data[['2Classes']]

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

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% Support Vector Machines

from sklearn import svm

SVM = svm.SVC(kernel = "linear", C=1.3)
SVM.fit(X, y.values.ravel())
cv_SVM = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
scores_SVM = cross_val_score(SVM, X, y.values.ravel(), scoring="roc_auc", cv=cv_SVM)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_SVM.mean(), scores_SVM.std()))
#%%
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=115, min_samples_split=5, random_state=50).fit(X, y.values.ravel())

cv_RF = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc", cv=cv_RF)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_RF.mean(), scores_RF.std()))

#%% Neural Network -- MLP Classifier

from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='sgd', alpha=4e-6, activation = 'logistic',
                   hidden_layer_sizes=(100,), random_state=1, max_iter=100000)

MLP.fit(X,y.values.ravel())
cv_MLP = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
scores_MLP = cross_val_score(MLP, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_MLP)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_MLP.mean(), scores_MLP.std()))

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

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X,y.values.ravel())

cv_gnb = ShuffleSplit(n_splits=10, test_size=0.25, random_state=50)
scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_gnb)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_gnb.mean(), scores_gnb.std()))
#%% Logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(penalty="l2",C=2, random_state=50, max_iter=1000).fit(X, y.values.ravel())

cv_LR = ShuffleSplit(n_splits=10, test_size=0.25, random_state=50)
scores_LR = cross_val_score(LR, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_LR)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_LR.mean(), scores_LR.std()))

#%% AdaBoost

from sklearn.ensemble import AdaBoostClassifier

AB = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=50, learning_rate=0.1, random_state=0).fit(X, y.values.ravel())

cv_AB = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_AB)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_AB.mean(), scores_AB.std()))


#%%

# --- SHAP values

import shap

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X)

# Plot variance importance
shap.summary_plot(shap_values, X, plot_type = 'bar')

#Plot impact on the model
shap.summary_plot(shap_values, X)

# Plot force graph
X_validation = validation[['API', 'Excipient1', 'Excipient2', 'Excipient3', 'Excipient4','x10', 'x50', 'x90', 'SMD', 'a10', 'a50', 'a90', 's10', 's50', 's90', 'BD(g/ml)']]
y_vallidation = validation[['2Classes']]


X_output = X_validation
X_output.loc[:,'predict'] = np.round(clf.predict(X_output),2)

# Randomly pick some observations
random_picks = np.arange(1,33,3) # Every 10 rows
S = X_output.iloc[random_picks]
# Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.
shap.initjs()

# Write in a function
def shap_plot(j):
    explainerModel = shap.TreeExplainer(clf)
    shap_values_Model = explainerModel.shap_values(S)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]], matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(0)

