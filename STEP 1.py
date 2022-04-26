import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#%%
path = r'C:\Users\ftb19213\Desktop\PhD\2022\Data\SEA'
file = r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\Model_A.csv'
validation= r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\External_data_validation.csv'
data = pd.read_csv(file)
validation = pd.read_csv(validation)

#%%

# -- DEfining variables

X = data.drop(['FFc', 'Class', '2Classes', 'Material'], axis =1)
data[['2Classes']] = data[['2Classes']].astype('category')
y = data[['2Classes']]


#%% KNN classifier
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier()
#kNN = KNeighborsClassifier(algorithm="brute", n_neighbors=11, metric="mahalanobis", metric_params={'VI': np.cov(X)})
kNN.fit(X,y.values.ravel())

#%% -- Evaluate kNN

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

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier()
GB.fit(X_train, y_train.values.ravel())

cv_GB = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
scores_GB = cross_val_score(GB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_GB)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_GB.mean(), scores_GB.std()))

#%%External validation

ext_val_X = validation.drop(['FFc', 'Class', '2Classes', 'Material', '1/ffc', 'log ffc'], axis =1)
ext_val_y = MLP.predict(ext_val_X)
ext_val_y

#%%

probability = MLP.predict_proba(ext_val_X)
probability

#%% -- SHAP values SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output
# of any machine learning model. It connects optimal credit allocation with local explanations using the 
#classic Shapley values from game theory and their related extensions.

# --- First, split into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

#%%
# -- import SHAP
import shap
shap.initjs()

explainer = shap.Explainer(GB)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)

#%% Froce plot

shap.initjs()

def shap_plot(j):
    explainer = shap.Explainer(GB)
    shap_values = explainer.shap_values(X_test)
    p = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(0)
