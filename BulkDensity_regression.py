# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:25:20 2022

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

#%%
path = r'C:\Users\ftb19213\Desktop\PhD\2022\Data\IMPORTANT DATASETS'
file = r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\Model_A.csv'
validation= r'C:\Users\ftb19213\Desktop\PhD\2021\Data\IMPORTANT DATASETS\External_data_validation.csv'
data = pd.read_csv(file)
validation = pd.read_csv(validation)

#%%
# ---------------------- DATA VISUALISATION -------------------------------
ax = data.hist(column='BD', bins=4, grid=False, figsize=(12,8),
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
    x.set_xlabel("Bulk density (g/ml)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Number of powders", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#%%
data_corr = data.drop(['Class', '2Classes','2Classes_B', 'Excipient1 concentration',
                       'Excipient2 concentration', 'Excipient3 concentration', 'Excipient4 concentration', 'Material',
                       'FFc'], axis =1)
cor = data_corr.corr().abs()
print(cor)
sns.heatmap(cor)
plt.show()

#%%
import seaborn as sns

selected_attributes = data[['PSD D10', 'PSD D50', 'D90',
       'Sphericity D10', 'Sphericity D90', 'Aspect ratio D10', 'BD']]

sns.pairplot(selected_attributes, diag_kind='kde')

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

# -- DEfining variables

X = data.drop(['FFc', 'Class', '2Classes', 'Material','2Classes_B', 'BD'], axis =1)
y = data[['BD']]

#%%
# --- First, split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=None, shuffle = True)


#%% 
#------------------ PLS --------------------------------

from sklearn.cross_decomposition import PLSRegression

pls2 = PLSRegression(n_components=2)
pls2.fit(X_train, y_train)

#%%
y_pred = pls2.predict(X)

#%%

pls2_score = pls2.score(X_test, y_test)
pls2_mae = mean_absolute_error(y_test.values.ravel(), pls2.predict(X_test))
print('PLS score: {:.2f}'.format(pls2_score))
print("PLS MAE on test set: {:.2f}".format(pls2_mae))

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## -- Select only PSD bins
#features = data.iloc[:,np.r_[10:110]]
features = data.drop(['Class', '2Classes', 'Material','2Classes_B', 'BD'], axis =1)

## Separating out the features
x = features.values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, svd_solver='full')
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1','principal component 2'])

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], alpha=0.3, label="samples")
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)
plt.legend()
plt.show()

#%% -- Linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

LR = LinearRegression().fit(X_train, y_train.values.ravel())

print("LR Score: %.3f" % LR.score(X_test, y_test.values.ravel()))

# Create the mean squared error
#
mse = mean_squared_error(y_test.values.ravel(), LR.predict(X_test))
print("Linear regression MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), LR.predict(X_test), squared=False)
print("Linear regression RMSE on test set: {:.2f}".format(rmse))

mae = mean_absolute_error(y_test.values.ravel(), LR.predict(X_test))
print("Linear regression MAE on test set: {:.2f}".format(mae))


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

from sklearn.ensemble import GradientBoostingRegressor

GB = GradientBoostingRegressor()
GB.fit(X_train, y_train.values.ravel())

print("GB R2: %.3f" % GB.score(X_test, y_test.values.ravel()))

mse = mean_squared_error(y_test.values.ravel(), GB.predict(X_test))
print("GB MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), GB.predict(X_test), squared=False)
print("GB RMSE on test set: {:.2f}".format(rmse))

GBmae = mean_absolute_error(y_test.values.ravel(), GB.predict(X_test))
print("GB MAE on test set: {:.2f}".format(GBmae))

#%%

from catboost import CatBoostRegressor

CatBoost = CatBoostRegressor().fit(X_train, y_train.values.ravel())

print("CatBoost R2: %.3f" % CatBoost.score(X_test, y_test.values.ravel()))

mse = mean_squared_error(y_test.values.ravel(), CatBoost.predict(X_test))
print("CatBoost MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), CatBoost.predict(X_test), squared=False)
print("CatBoost RMSE on test set: {:.2f}".format(rmse))

mae = mean_absolute_error(y_test.values.ravel(), CatBoost.predict(X_test))
print("CatBoost MAE on test set: {:.2f}".format(mae))

#%%

from xgboost import XGBRegressor

XGBR = XGBRegressor().fit(X_train, y_train.values.ravel())

print("XGBR R2: %.3f" % XGBR.score(X_test, y_test.values.ravel()))

mse = mean_squared_error(y_test.values.ravel(), XGBR.predict(X_test))
print("XGBR MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), XGBR.predict(X_test), squared=False)
print("XGBR RMSE on test set: {:.2f}".format(rmse))

mae = mean_absolute_error(y_test.values.ravel(), XGBR.predict(X_test))
print("XGBR MAE on test set: {:.2f}".format(mae))

#%%
from sklearn.ensemble import AdaBoostRegressor

AB = AdaBoostRegressor().fit(X_train, y_train.values.ravel())

print("AB R2: %.3f" % AB.score(X_test, y_test.values.ravel()))

mse = mean_squared_error(y_test.values.ravel(), AB.predict(X_test))
print("AB MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), AB.predict(X_test), squared=False)
print("AB RMSE on test set: {:.2f}".format(rmse))

ABmae = mean_absolute_error(y_test.values.ravel(), AB.predict(X_test))
print("AB MAE on test set: {:.2f}".format(ABmae))

#%%

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SVR = make_pipeline(StandardScaler(), SVR()).fit(X_train, y_train.values.ravel())

print("SVR R2: %.3f" % SVR.score(X_test, y_test.values.ravel()))

mse = mean_squared_error(y_test.values.ravel(), SVR.predict(X_test))
print("SVR MSE on test set: {:.2f}".format(mse))

rmse = mean_squared_error(y_test.values.ravel(), SVR.predict(X_test), squared=False)
print("SVR RMSE on test set: {:.2f}".format(rmse))

SVRmae = mean_absolute_error(y_test.values.ravel(), SVR.predict(X_test))
print("AB MAE on test set: {:.2f}".format(SVRmae))

#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Algorithms = ['RF','GB', 'AB', 'PLS']
Scores = [RFmae,GBmae,ABmae,pls2_mae]

ax.bar(Algorithms,Scores)
plt.title("MAE Comparison")
plt.xlabel('Models')
plt.ylabel('MAE')

plt.show()

#%% Feature importance
import seaborn as sns

# Add 'Wall friction angle - PHIE [°]', 'PHIE_class'   when all data file
# Add '3Classes' when needed
parameters_list =list(data.columns.drop(['FFc', 'Class', '2Classes', 'Material', '2Classes_B']))
parameters_list=np.array(parameters_list)
print(parameters_list)


imp_score = pd.Series(RF.feature_importances_, index=parameters_list).sort_values(ascending=False)
print(imp_score[:5])


#Visualising the importance score
sns.barplot(x=imp_score[:5], y=imp_score.index[:5])
# Add labels to your graph
plt.xlabel('Features')
plt.ylabel('Feature Importance Score')
plt.title("Visualising Important Features")
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

shap.summary_plot(shap_values, X, plot_type ='bar')

#%%
shap.initjs()

#shap.dependence_plot('PSD D10', shap_values ,X, interaction_index="PSD D50")
shap.dependence_plot('PSD D10', shap_values, X, interaction_index="PSD D10")

#%%External validation
#Drop variables from validation set (the same as for the training)


validation = validation.drop(validation[to_drop],axis=1)
validation.head()

#%%
ext_val_X = validation.drop(['FFc', 'Class', '2Classes','2Classes_B', 'Material', '1/ffc', 'log ffc', 'BD'], axis =1)
actual_ext_val_y = validation[['BD']]

#%%
ext_val_y = RF.predict(ext_val_X)
ext_val_y

#%%

from matplotlib import pyplot

actual_ext_val_y = validation[['BD']]

pyplot.scatter(ext_val_y,actual_ext_val_y)
m, b = np.polyfit(ext_val_y,actual_ext_val_y,1)
plt.plot(ext_val_y, m*ext_val_y+b, color='blue')

plt.title("External validation")
plt.xlabel("Predicted FFc")
plt.ylabel("Actual FFc")
plt.show()

#%%
import scipy.stats as stats

actual = np.asarray(actual_ext_val_y)
actual = actual.reshape(1,8)
predicted = np.array(ext_val_y)

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.
    
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}
    
    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    
    """
    if ax is None:
        ax = plt.gca()
    
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax


x = actual[0,:]
y = predicted

# Modeling with Numpy
def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b) 

p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

# Statistics
n = y_test.size                                           # number of observations
m = p.size                                                 # number of parameters
dof = n - m                                                # degrees of freedom
t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands

# Estimates of Error in Data/Model
resid = y - y_model                           
chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

# Plotting --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
plt.xlabel('Actual BD')
plt.ylabel('Predicted BD')

# Data
ax.plot(
    x, y, "o", color="#b9cfe7", markersize=8, 
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
)
ax.plot(x,x,'k-')

# Fit
ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

x2 = np.linspace(np.min(x), np.max(x), 100)
y2 = equation(p, x2)

# Confidence Interval (select one)
plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
#plot_ci_bootstrap(x, y, resid, ax=ax)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
display = (0, 1)
anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
legend = plt.legend(
    [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
    [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
    loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
)  
frame = legend.get_frame().set_edgecolor("0.5")
   
# Prediction Interval
pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
ax.plot(x2, y2 + pi, "--", color="0.5")

#plt.show()

#%%

from sklearn.metrics import r2_score
r2 = r2_score(ext_val_y,actual_ext_val_y)
print('R2 score for external validation is: %.3f' % r2)
#%% -- SHAP values SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output
# of any machine learning model. It connects optimal credit allocation with local explanations using the 
#classic Shapley values from game theory and their related extensions.

# -- import SHAP
import shap
shap.initjs()

explainer = shap.Explainer(RF)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)

#%% Froce plot

shap.initjs()

def shap_plot(j):
    explainer = shap.Explainer(GB)
    shap_values = explainer.shap_values(ext_val_X)
    p = shap.force_plot(explainer.expected_value, shap_values[4,:], ext_val_X.iloc[4,:], matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(4)


#%%

# ------------------------  CLASSIFICATION ---------------------------------------

category = pd.cut(data.BD,bins=[0,0.5,1.229],labels=[0,1])
data.insert(3, 'DensityCategory', category)

#%%

X = data.drop(['Class', '2Classes', 'Material','2Classes_B', 'BD', 'DensityCategory', 'FFc'], axis =1)
data[['DensityCategory']] = data[['DensityCategory']].astype('category')
y = data[['DensityCategory']]


#%% KNN classifier
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=11)
#kNN = KNeighborsClassifier(algorithm="brute", n_neighbors=11, metric="mahalanobis", metric_params={'VI': np.cov(X)})
kNN.fit(X,y.values.ravel())

#%% -- Evaluate kNN

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(kNN, X, y.values.ravel(), scoring="roc_auc", cv=cv)
#scores = cross_val_score(kNN, X, y.values.ravel(), scoring="roc_auc_ovr", cv=cv)

print("kNN %0.3f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% Support Vector Machines

from sklearn import svm

SVM = svm.SVC(kernel = "linear")#, C=1.3)
SVM.fit(X, y.values.ravel())
cv_SVM = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_SVM = cross_val_score(SVM, X, y.values.ravel(), scoring="roc_auc", cv=cv_SVM)

print("SVM %0.3f accuracy with a standard deviation of %0.2f" % (scores_SVM.mean(), scores_SVM.std()))

#%%
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=200)#, min_samples_split=5
RF.fit(X, y.values.ravel())

cv_RF = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc", cv=cv_RF)
#scores_RF = cross_val_score(RF, X, y.values.ravel(), scoring="roc_auc_ovr_weighted", cv=cv_RF)

print("RF %0.3f accuracy with a standard deviation of %0.2f" % (scores_RF.mean(), scores_RF.std()))

#%% Neural Network -- MLP Classifier

from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='sgd', alpha=4e-6, activation = 'logistic',
                   hidden_layer_sizes=(100,), random_state=1, max_iter=100000)


MLP.fit(X,y.values.ravel())
cv_MLP = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
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

cv_gnb = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_gnb)
#scores_gnb = cross_val_score(gnb, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_gnb)

print("GNB %0.3f accuracy with a standard deviation of %0.2f" % (scores_gnb.mean(), scores_gnb.std()))
#%% Logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(penalty="l2",C=2, solver='liblinear',random_state=None, max_iter=1000).fit(X, y.values.ravel())

cv_LR = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_LR = cross_val_score(LR, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_LR)


print("LR %0.3f accuracy with a standard deviation of %0.2f" % (scores_LR.mean(), scores_LR.std()))

#%% AdaBoost

from sklearn.ensemble import AdaBoostClassifier

AB = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=50,
                        learning_rate=0.1, random_state=0).fit(X, y.values.ravel())


cv_AB = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_AB)
#scores_AB = cross_val_score(AB, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_AB)

print("AB %0.3f accuracy with a standard deviation of %0.2f" % (scores_AB.mean(), scores_AB.std()))

#%%

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier()
GB.fit(X, y.values.ravel())
#GB.fit(X_train, y_train.values.ravel())

cv_GB = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
scores_GB = cross_val_score(GB, X, y.values.ravel(), scoring = "roc_auc",  cv=cv_GB)
#scores_GB = cross_val_score(GB, X, y.values.ravel(), scoring = "roc_auc_ovr_weighted",  cv=cv_GB)

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
from sklearn.metrics import ConfusionMatrixDisplay

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        GB,
        X,
        y,
        #display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


#%%
from sklearn.decomposition import PCA

## -- Select only PSD bins
#features = data.iloc[:,np.r_[10:110]]
features = data.drop(['Class', '2Classes', 'Material','2Classes_B', 'BD', 'DensityCategory' ], axis =1)

## Separating out the features
x = features.values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, svd_solver='full')
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1','principal component 2'])

finalDf = pd.concat([principalDf, data[['DensityCategory']]], axis = 1)

#Visualising

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['DensityCategory'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 60)
ax.legend(targets)
ax.grid()
#pca.explained_variance_ratio_
print ( "Components = ", pca.n_components_ , ";\nTotal explained variance = ",
      round(pca.explained_variance_ratio_.sum(),2)  )


#%% Feature importance
import seaborn as sns


# Add '3Classes' when needed
parameters_list =list(data.columns.drop(['FFc', 'Class', '2Classes','2Classes_B', 'Material', '2Classes_B']))
parameters_list=np.array(parameters_list)
print(parameters_list)


imp_score = pd.Series(RF.feature_importances_, index=parameters_list).sort_values(ascending=False)
print(imp_score[:5])


#Visualising the importance score
sns.barplot(x=imp_score[:5], y=imp_score.index[:5])
# Add labels to your graph
plt.xlabel('Features')
plt.ylabel('Feature Importance Score')
plt.title("Visualising Important Features")
plt.show()

#%%External validation
#Drop variables from validation set (the same as for the training)

category = pd.cut(validation.BD,bins=[0,0.5,1.229],labels=[0,1])
validation.insert(3, 'DensityCategory', category) 

#%%
validation = validation.drop(validation[to_drop],axis=1)
validation.head()

#%%
ext_val_X = validation.drop(['BD', 'Class', '2Classes','2Classes_B', 'Material', '1/ffc',
                             'log ffc', 'DensityCategory', 'FFc'], axis =1)
    
validation[['DensityCategory']] = validation[['DensityCategory']].astype('category')
actual_ext_val_X = validation[['DensityCategory']]

#%%
ext_val_y = RF.predict(ext_val_X)
ext_val_y

#%%

probability = RF.predict_proba(ext_val_X)
probability

#%% -- SHAP values SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output
# of any machine learning model. It connects optimal credit allocation with local explanations using the 
#classic Shapley values from game theory and their related extensions.

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
    shap_values = explainer.shap_values(ext_val_X)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], ext_val_X, show = False)
    plt.savefig('tmp.png')
    plt.close()
    return(p)

# --- Force plot that will get the result of the first row of X_test    
shap_plot(7)

#%%

shap.dependence_plot('FFc', shap_values[1], X, interaction_index="FFc")