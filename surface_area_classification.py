# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:01:31 2022

@author: ftb19213
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


path = r'C:\Users\ftb19213\Desktop\PhD\2022\Data\SEA'
file = r'C:\Users\ftb19213\Desktop\PhD\2022\Data\SEA\SurfaceAreaClasses.csv'
data = pd.read_csv(file, encoding= 'unicode_escape')

#%%

# ---------------------- DATA VISUALISATION -------------------------------
ax = data.hist(column='SurfaceArea', bins=10, grid=False, figsize=(12,8),
             color='#86bf91', zorder=2, rwidth=0.9)
             

ax = ax[0]
for x in ax:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on",
                  left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Surface energy (mJ/m\u00b2)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Number of powders", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#%%
data_corr = data.drop(['Excipient1', 'Excipient2', 'Excipient3', 'Excipient4',
                       'SpecificSE', 'SEcom', 'DSEat0%', 'DSEat3%', 'DSEat5%',
                       'DSEat10%'],
                      axis =1)
cor = data_corr.corr().abs()
print(cor)
sns.heatmap(cor)
plt.show()

#%%
 
#Increase the size of heatmap

plt.figure(figsize=(16, 6))

# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation 
#to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(data_corr.corr(), vmin=-1, vmax=1, annot=True)

# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

#%%

# filter PCC > 0.9
upper_tri = cor.where(np.triu(np.ones(cor.shape),k=1).astype(np.bool))
print(upper_tri)

#%%
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(to_drop)

#%%
data=data.drop(data[to_drop],axis=1)
data.head()

#%%

category = pd.cut(data.SurfaceArea,bins=[0,0.65, 2.77],labels=[0,1])
data.insert(22, 'SurfaceAreaCategory', category)


#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## -- Select only PSD bins
#features = data.iloc[:,np.r_[10:110]]
features = data.drop(['Material', 'SurfaceAreaCategory','SurfaceArea',
                      'SpecificSE','SEcom', 'DSEat0%', 'DSEat3%', 'DSEat5%','DSEat10%' ], axis =1)

## Separating out the features
x = features.values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, svd_solver='full')
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1','principal component 2'])

finalDf = pd.concat([principalDf, data[['SurfaceAreaCategory']]], axis = 1)

#Visualising

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['SurfaceAreaCategory'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 60)
ax.legend(targets)
ax.grid()
#pca.explained_variance_ratio_
print ( "Components = ", pca.n_components_ , ";\nTotal explained variance = ",
      round(pca.explained_variance_ratio_.sum(),2)  )

#%% Defining variables

# 'SurfaceArea',
X = data.drop(['Material','SurfaceAreaCategory','SpecificSE','SurfaceArea',
               'SEcom', 'DSEat0%', 'DSEat3%', 'DSEat5%', 'DSEat10%','FFc' ], axis =1)
data[['SurfaceAreaCategory']] = data[['SurfaceAreaCategory']].astype('category')
y = data[['SurfaceAreaCategory']]

#%% KNN classifier
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=5)
#kNN = KNeighborsClassifier(algorithm="brute", n_neighbors=11, metric="mahalanobis", metric_params={'VI': np.cov(X)})
kNN.fit(X,y.values.ravel())

#%% -- Evaluate kNN

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores = cross_val_score(kNN, X, y.values.ravel(), scoring="roc_auc", cv=cv)
#scores = cross_val_score(kNN, X, y.values.ravel(), scoring="roc_auc_ovr_weighted", cv=cv)

print("kNN %0.3f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% Support Vector Machines

from sklearn import svm

SVM = svm.SVC()#kernel = "linear", C=1.3)
SVM.fit(X, y.values.ravel())
cv_SVM = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_SVM = cross_val_score(SVM, X, y.values.ravel(), scoring="roc_auc", cv=cv_SVM)

print("SVM %0.3f accuracy with a standard deviation of %0.2f" % (scores_SVM.mean(), scores_SVM.std()))

#%%
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()#n_estimators=115, min_samples_split=5
RF.fit(X, y.values.ravel())

cv_RF = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc", cv=cv_RF)
acc_scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="accuracy", cv=cv_RF)
precision_scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="precision", cv=cv_RF)

#scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc_ovr_weighted", cv=cv_RF)

print("RF %0.3f AUC with a standard deviation of %0.2f" % (scores_RF.mean(), scores_RF.std()))
print("RF %0.3f Accuracy with a standard deviation of %0.2f" % (acc_scores_RF.mean(), acc_scores_RF.std()))
print("RF %0.3f Precision with a standard deviation of %0.2f" % (precision_scores_RF.mean(), precision_scores_RF.std()))

#%% Neural Network -- MLP Classifier

from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='sgd', alpha=4e-6, activation = 'logistic',
                   hidden_layer_sizes=(100,), random_state=1, max_iter=100000)


MLP.fit(X,y.values.ravel())
cv_MLP = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_MLP = cross_val_score(MLP, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_MLP)
#scores_MLP = cross_val_score(MLP, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_MLP)

print("MLP %0.3f accuracy with a standard deviation of %0.2f" % (scores_MLP.mean(), scores_MLP.std()))

#%% Confusion matrix for MLP Classifier

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(RF, X, y.values.ravel(), cv=10)
conf_mat = confusion_matrix(y, y_pred)
conf_mat

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X,y.values.ravel())

cv_gnb = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "accuracy",  cv=cv_gnb)
#scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_gnb)

print("GNB %0.3f accuracy with a standard deviation of %0.2f" % (scores_gnb.mean(), scores_gnb.std()))
#%% Logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(penalty="l2",C=2, solver='liblinear',random_state=50, max_iter=1000).fit(X, y.values.ravel())

cv_LR = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_LR = cross_val_score(LR, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_LR)


print("LR %0.3f accuracy with a standard deviation of %0.2f" % (scores_LR.mean(), scores_LR.std()))

#%% AdaBoost

from sklearn.ensemble import AdaBoostClassifier

AB = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=50,
                        learning_rate=0.1, random_state=0).fit(X, y.values.ravel())


cv_AB = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_AB)
#scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_AB)

print("AB %0.3f accuracy with a standard deviation of %0.2f" % (scores_AB.mean(), scores_AB.std()))

#%%

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier()
GB.fit(X, y.values.ravel())
#GB.fit(X_train, y_train.values.ravel())

cv_GB = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
#scores_GB = cross_val_score(GB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_GB)
scores_GB = cross_val_score(GB, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_GB)

print("GB %0.3f accuracy with a standard deviation of %0.2f" % (scores_GB.mean(), scores_GB.std()))

#%%

# Make bar plot for comparison of the performance of the algorithm
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Algorithms = ['kNN','SVM', 'RF', 'MLP NN', 'NB','LR', 'AB', 'GB']
Scores = [scores.mean(),scores_SVM.mean(), scores_RF.mean(), scores_MLP.mean(), 
          scores_gnb.mean(), scores_LR.mean(), scores_AB.mean(), scores_GB.mean()]

ax.bar(Algorithms,Scores)
plt.title("ROC_AUC Performance Comparison")
plt.xlabel('Models')
plt.ylabel('ROC_AUC')

plt.show()

#%%

# -- import SHAP
import shap
shap.initjs()

#X does not have valid feature names, but MLPClassifier was fitted with feature names
#Using 112 background data samples could cause slower run times. Consider using shap.sample(data, K)
# or shap.kmeans(data, K) to summarize the background as K samples.

explainer = shap.Explainer(RF)
shap_values = explainer.shap_values(X)
shap_obj=explainer(X)

shap.summary_plot(shap_values, X)

shap.summary_plot(shap_values[0], X)
shap.summary_plot(shap_values[1], X)
#shap.summary_plot(shap_values[2], X)

shap.summary_plot(shap_values[0], X, plot_type ='bar')
shap.summary_plot(shap_values[1], X, plot_type ='bar')
#shap.summary_plot(shap_values[2], X, plot_type ='bar')



#%% Froce plot (bettwe for regression)
shap.initjs()

def shap_plot(j):
    explainer = shap.Explainer(RF)
    shap_values = explainer.shap_values(X)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], X, show = False)
    plt.savefig('tmp.png')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(7)

#%%

shap.dependence_plot('PSD D90', shap_values[1], X, interaction_index="PSD D90")

#%%
import seaborn as sns

selected_attributes = data[['PSD D10', 'PSD D50', 'PSD D90',
       'Sphericity D90', 'Aspect ratio D90', 'BD (g/ml)',
       'SurfaceArea']]

sns.pairplot(selected_attributes, diag_kind='kde')

#%%
import matplotlib.pyplot as plt
sns.heatmap(data[['SpecificSE', 'Sphericity D90', 'PSD D50']].corr(), cmap='Blues', annot=True)
plt.show()

#%%
# -------------------------------------- EXTERNAL VALIDATION -----------------------------------------

validation= r'C:\Users\ftb19213\Desktop\PhD\2022\Data\SEA\External_validation_surface_area.csv'
validation = pd.read_csv(validation)

#%%

category = pd.cut(validation.SpecificSE,bins=[0,6.97,16.81],labels=[0,1])
validation.insert(3, 'SurfaceEnergyCategory', category)

#%%
validation = validation.drop(validation[to_drop],axis=1)
validation.head()

ext_val_X = validation.drop(['Material','SurfaceEnergyCategory', 'SurfaceArea','SpecificSE',
               'SE(com)', 'DSEat0%', 'DSEat3%', 'DSEat5%', 'DSEat10%'], axis =1)
    
validation[['SurfaceEnergyCategory']] = validation[['SurfaceEnergyCategory']].astype('category')
actual_ext_val_y = validation[['SurfaceEnergyCategory']]

#%%
ext_val_y = RF.predict(ext_val_X)
ext_val_y

#%%

probability = RF.predict_proba(ext_val_X)
probability

#%%

# ---------------------- DATA VISUALISATION -------------------------------
ax = data.hist(column='SpecificSE', bins=10, grid=False, figsize=(12,8),
             color='#86bf91', zorder=2, rwidth=0.9)
             

ax = ax[0]
for x in ax:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on",
                  left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Specific surface energy", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Number of powders", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#%%
    # --------------------------------- REGRESSION --------------------------------------
    
    
X = data.drop(['Material', 'SurfaceArea','SpecificSE',
               'SEcom', 'DSEat0%', 'DSEat3%', 'DSEat5%', 'DSEat10%' ], axis =1)
y = data[['SpecificSE']]

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=None, shuffle = True)

#%%

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor().fit(X_train, y_train.values.ravel())

print("RF R2: %.3f" % RF.score(X_test, y_test.values.ravel()))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test.values.ravel(), RF.predict(X_test))
print("RF MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), RF.predict(X_test), squared=False)
print("RF RMSE on test set: {:.2f}".format(rmse))

RFmae = mean_absolute_error(y_test.values.ravel(), RF.predict(X_test))
print("RF MAE on test set: {:.2f}".format(RFmae))

#%%

# -- import SHAP
import shap
shap.initjs()

#X does not have valid feature names, but MLPClassifier was fitted with feature names
#Using 112 background data samples could cause slower run times. Consider using shap.sample(data, K)
# or shap.kmeans(data, K) to summarize the background as K samples.

explainer = shap.Explainer(RF)
shap_values = explainer.shap_values(X_train)
shap_obj=explainer(X_train)

shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X_train, plot_type ='bar')




#%% Froce plot (bettwe for regression)
shap.initjs()

def shap_plot(j):
    explainer = shap.Explainer(RF)
    shap_values = explainer.shap_values(X_test)
    p = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:],
                        matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(0)

#%%

shap.dependence_plot('PSD D50', shap_values, X_train, interaction_index="PSD D50")

#%%
import seaborn as sns

selected_attributes = data[['SpecificSE', 'Sphericity D90', 'PSD D50']]

sns.pairplot(selected_attributes, diag_kind='kde')