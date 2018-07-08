
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from __future__ import division


# A more complete description of the data, including the source, can be found in the git README.md file.
# 
# Labs and vitals categories were previously discretized from their raw values using knn into 5 clusters; no code book was available for interpretation,so removed from model.
# 
# AbxUTI was removed as it is correlated to all cases doctor diagnosed as UTI.
# 
# Dummy variables were created for all string categorical features.
# 
# All features with > 20% not_recorded values were dropped.
# 
# drop PATID, ID (in the future, retain PATID to be able to compare predictions from test set back to doctor assessment).
# 
# drop ua_bacteria as it is correlated with diagnosis and not useful for clinical prediction
# 
# recode from Yes/No to 1/0: in this order so true_dx is the last column for y_pred
# 
# prior_history outpat_meds abxUTI to abxUTI (yes/no) UTI_diag to dr_dx UCX_abnormal to true_dx
# 
# outcome variable and comparison outcome variable (doctor diagnosis)
# 
# data["dr_dx"] = pd.Series(np.where(data["UTI_diag"].values == "Yes", 1,0),data.index) data["true_dx"] = pd.Series(np.where(data["UCX_abnormal"].values == "yes", 1,0),data.index)
# 
# recoded everything to numeric: for pH, had to replace NaN values with imputed mean (ran test to be sure mean didn't differ between outcome types)
# 
# Balanced data set to achieve 50% true_dx 0 and 1.  This reduced the data set from 80,000 observations to 40,000.
# 
# Saved the cleaned data for the model: mod2.to_csv("uti_recoded_model2.tsv", sep='\t')
# 
# I looked at the correlations here: mod2_corr = mod2.corr() resamp_corr = resamp.corr() then made a list of those with corr > 70%
# 
# indices = np.where(mod2_corr > 0.7) indices = [(mod2_corr.index[x], mod2_corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]
# 
# resamp_indices = np.where(resamp_corr > 0.7) resamp_indices = [(resamp_corr.index[x], resamp_corr.columns[y]) for x, y in zip(*resamp_indices) if x != y and x < y] resamp_indices
# 
# A separate notebook was created to explore and annotate building a model based on extracting only female observations from the data set.  This data set was not balanced. (A separate notebook was run for a female-only, balanced data set, but the observations were only 14K each outcome).  Women get UTI much more frequently than men, so it is of interest to know whether the predictors or feature importance changes if we consider only women.  According to that set of operations, the features are very similar but the strength of some of the predictors is different.  This can be compared in the slide deck at:
# bit.ly/Sheila_Adams-Sapper_demo
# 
# Next steps:  Try to weight the classes rather than dropping observations to balance the data set.

# In[2]:


mod2 = pd.read_csv('uti_recoded_model2.tsv', sep = '\t')
mod2 = mod2.iloc[:,1:]


# After the first pass of cleaning/pre-processing the data I noticed a problem with specific gravity values that occurred below and above physiolog. normal:

# In[65]:


mod2['ua_spec_grav'].value_counts()


# find the rows where specific gravity is below physiol. norms between 1 - 1.030, allow for abnormal test results due to dehydration and kidney dysfunction:

# In[81]:


sg_low = mod2[mod2['ua_spec_grav']< 1]
sg_high = mod2[mod2['ua_spec_grav']> 1.5]


# In[70]:


sg_high


# In[77]:


sg_low


# re-code as boolean to be able to drop the extreme data observations from the data set

# In[ ]:


sg_low = mod2['ua_spec_grav']< 1
sg_high = mod2['ua_spec_grav']> 1.5


# drop rows where specific gravity is below/above physiol. norms between 1 - 1.030

# In[84]:


mod2.drop(mod2[sg_low].index, axis=0,inplace=True)
mod2.drop(mod2[sg_high].index, axis=0,inplace=True)


# In[85]:


mod2.shape


# Balance the data by subsampling from the more frequent no UTI (0) outcome category

# In[86]:


resamp = mod2.drop(mod2[mod2['true_dx'] == 0].sample(frac=0.70).index)
resamp.shape


# Scale the continuous variables

# In[87]:


age =(resamp["age"]- resamp["age"].min())/(resamp["age"].max() - resamp["age"].min())
resamp["age"] = age

ph =(resamp["ua_ph"]- resamp["ua_ph"].min())/(resamp["ua_ph"].max() - resamp["ua_ph"].min())
resamp["ua_ph"] = ph

spec_grav =(resamp["ua_spec_grav"]- resamp["ua_spec_grav"].min())/(resamp["ua_spec_grav"].max() - resamp["ua_spec_grav"].min())
resamp["ua_spec_grav"] = spec_grav


# In[88]:


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].hist('ua_ph', data=resamp,label='pH', bins=10, alpha=0.6)
axes[0].set_title('pH distrib.', fontsize=8)
axes[1].hist('ua_spec_grav', data=resamp,label='specific gravity', bins=10, alpha=0.6)
axes[1].set_title('specific gravity distrib.', fontsize=8)
plt.show()


# In[89]:


pH_byDx = resamp.groupby("true_dx")["ua_ph"]
sg_byDx = resamp.groupby("true_dx")["ua_spec_grav"]
# fig, axes = plt.subplots(2,1,1, figsize=(12, 4))
plt.subplot(1, 2, 1)
pH_byDx.plot(kind='hist', figsize=[10,4], alpha=.4, legend=True, title = "pH distrib. by UTI diagnosis (bal. data set)")
plt.subplot(1, 2, 2)
sg_byDx.plot(kind='hist', figsize=[10,4], alpha=.4, legend=True, title = "Specific gravity distrib. by UTI diagnosis (bal. data set)")
plt.show()


# Look at the doctors' assessment performance and look at the differences in means by outcome

# In[5]:


drdx_v_true = pd.crosstab(resamp["true_dx"], resamp["dr_dx"])
drdx_v_true


# In[90]:


resamp.groupby('true_dx').mean()


# Look at the age distribution after balancing the data set

# In[49]:


age_byDx = resamp.groupby("true_dx")["age"]
age_byDx.plot(kind='hist', figsize=[6,4], alpha=.4, legend=True, title = "Age distribution by UTI diagnosis (balanced data set)") # alpha for transparency
plt.savefig('age_distrib_bal.pdf')


# True diagnosis by gender:

# In[101]:


print((resamp['gender_Female'].sum()/len(resamp)),(resamp['gender_Male'].sum()/len(resamp)))


# In[96]:


female_v_true = pd.crosstab(resamp["true_dx"], resamp["gender_Female"])
female_v_true


# In[97]:


male_v_true = pd.crosstab(resamp["true_dx"], resamp["gender_Male"])
male_v_true


# Make the first split for training and the hold-out test set

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

X = resamp.iloc[:, 0:243].values
y = resamp.iloc[:, 244].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 11)


# For Recursive Feature Elimination, look at the grid curve for accuracy tradeoff for a smaller set of features
# set the C regularization parameter very high so it does not apply weighting for RFE (The sklearn LogisticRegression has an L2 penalty default that cannot be turned off - unless one wants to select L1)

# In[52]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


# Use X_train created above from 80/20 split of resamp data set

# Create the RFE object and compute a cross-validated score.
logistic = linear_model.LogisticRegression(C=10**6)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=logistic, step=1, cv=StratifiedKFold(3),
              scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('RFECV_grid_largeC.png')
plt.show()


# In[53]:


rfecv.grid_scores_[0:30]


# Make a second split in the training set to allow for a separate validation set prior to final testing on the hold-out test set.

# In[8]:


from sklearn.cross_validation import train_test_split
X_train_rfe, X_val_rfe, y_train_rfe, y_val_rfe = train_test_split(X_train, y_train, test_size = 0.20, random_state = 1)


# RFE originally run with 30 features, but had a very high number of 'not_reported' variables. The final test was run with 16, which is the transition point in the GridSearch curve beyond which not much additional accuracy is obtained.

# In[48]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10**6)

rfe = RFE(logreg,16)
rfe = rfe.fit(X_train_rfe, y_train_rfe)
print(rfe.support_)
print(rfe.ranking_)


# In[50]:


rfe.score(X_train_rfe, y_train_rfe)


# In[51]:


cols = []
for i,val in enumerate(rfe.support_):
    if val == True:
        cols.append(resamp.columns[i])
cols


# Make the training set into a dataframe to be able to pass the new column names to the orig X_train matrix

# In[9]:


train_cols = resamp.columns[0:(len(resamp.columns)-2)] #to remove dr_dx, true_dx, the last 2 columns in resamp
X_train_rfe_df = pd.DataFrame(X_train_rfe)
X_train_rfe_df.columns = train_cols
X_train_rfe_df.head()


# Proceed to Logistic Regression with L1 regularization penalty to compare feature output

# Perform GridSearch for optimum alpha.  Ideal was given as 0.001, but exploration done with 0.01, 0.005, and 0.001 showed that 0.005 was preferable in terms of trade-off between feature relevance and model accuracy.

# In[45]:


from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


lasso = Lasso(random_state=0, max_iter=1e5)
alphas = np.logspace(-3, -0.5, 10)

tuned_parameters = [{'alpha': alphas}]
n_folds = 3

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])


# In[19]:


from sklearn import linear_model

logl1_005 = LogisticRegression(random_state=123, penalty='l1', solver='liblinear', C=0.005)
logl1_005.fit(X_train_rfe,y_train_rfe)

print(logl1_005.coef_)
print(logl1_005.intercept_)


# Notebook output differs slight when re-run:  vitamins and ua_protein_negative were not previously in the list except with regularization parameter higher than 0.005.  Will remove for consistency of past reports.

# In[20]:


logl1_005_coef = np.nonzero(logl1_005.coef_)[1]
for i in logl1_005_coef:
    print(resamp.columns[i])


# In[12]:


l1_cols = ['abx','ua_bili_negative','ua_blood_negative','ua_clarity_clear','ua_leuk_large','ua_leuk_negative',
           'ua_nitrite_negative','ua_nitrite_positive','ua_urobili_negative','gender_Female','gender_Male',
           'employStatus_Retired','insurance_status_Medicare']


# In[13]:


X_l1 = X_train_rfe_df[l1_cols] # new X with reduced features
y_l1 = y_train_rfe.ravel()


# compare coeficients to logistic model without regularization parameters built in

# In[14]:


import statsmodels.api as sm
logit_model=sm.Logit(y_l1,X_l1.astype(float))
result=logit_model.fit()
print(result.summary())


# In[21]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(random_state=123, penalty='l1', solver='liblinear', C=0.005)
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_l1, y_l1, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# Prepare the validation set for model testing

# In[22]:


train_cols = resamp.columns[0:(len(resamp.columns)-2)]
X_val_rfe_df = pd.DataFrame(X_val_rfe)
X_val_rfe_df.columns = train_cols

y_val_test = y_val_rfe.ravel()
X_val_test_l1 = X_val_rfe_df[l1_cols]
X_val_test_l1.shape


# In[25]:


y_l1_train = pd.DataFrame(y_l1)


# Fit the model with the smaller feature set and get the prediction from the validation set.

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg_l1 = LogisticRegression(random_state=123, penalty='l1', solver='liblinear', C=0.005)
logreg_l1.fit(X_l1, y_l1)
print(logreg_l1.intercept_, logreg_l1.coef_)


# In[29]:


y_pred_l1 = logreg_l1.predict(X_val_test_l1)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_l1.score(X_val_test_l1, y_val_test)))


# In[30]:


from sklearn.metrics import confusion_matrix
cm_l1_final = confusion_matrix(y_val_test, y_pred_l1)
print(cm_l1_final)


# In[31]:


cm_l1_val_set = cm_l1_final.astype('float') / cm_l1_final.sum(axis=1)[:, np.newaxis]
cm_l1_val_set


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_val_test, y_pred_l1))


# Test the model against the hold-out test set. First prepare the test set as df.

# In[33]:


train_cols = resamp.columns[0:(len(resamp.columns)-2)] #to eliminate true_dx
X_final_test_df = pd.DataFrame(X_test)
X_final_test_df.columns = train_cols
X_final_test_df.head()


# In[34]:


y_final_test = y_test
X_final_test = X_final_test_df[l1_cols]
X_final_test.shape


# This file was saved to pickle for the online tool.
# X_final_test.to_csv('x_test_l1_final.tsv', sep = '\t')

# In[37]:


y_pred_l1_final = logreg_l1.predict(X_final_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_l1.score(X_final_test, y_final_test)))


# In[38]:


from sklearn.metrics import confusion_matrix
confusion_matrix_l1 = confusion_matrix(y_final_test, y_pred_l1_final)
print(confusion_matrix_l1)


# In[39]:


cm_l1_transform = confusion_matrix_l1.astype('float') / confusion_matrix_l1.sum(axis=1)[:, np.newaxis]
cm_l1_transform


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_final_test, y_pred_l1_final))


# Plot the features by importance

# In[42]:


from pandas import Series, DataFrame

comm_names = ['prior hist. antibiotic use', 'bilirubin negative', 'blood in urine, negative', 'urine clear', 
              'large leukocyte count','leukocyte count negative', 'nitrite test, negative', 'nitrite test, positive',
              'urobilinogen, negative', 'female gender','male gender','employment: retired','medicare insurance']
plt.figure(figsize=(8,6))
plt.tight_layout()
predictors_final = comm_names
predictors_final = Series(logreg_l1.coef_[0,:],predictors_final).sort_values()
predictors_final.plot(kind='barh', title= 'Logistic(L1 Penalty) Model Coefficients', color='orange')
plt.savefig('l1_model_coef_balanced.png')


# In[43]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

#doctor pred
pred = resamp.iloc[:,243]
true = resamp.iloc[:,244]
fpr, tpr, thresh = metrics.roc_curve(true, pred)
auc = metrics.roc_auc_score(true, pred)
plt.plot(fpr,tpr,label="doctor pred, auc="+str(auc))

#L1 hold-out test set y_pred
pred = y_pred_l1_final
true = y_test
fpr, tpr, thresh = metrics.roc_curve(true, pred)
auc = metrics.roc_auc_score(true, pred)
plt.plot(fpr,tpr,label="L1 pred (test set), auc="+str(auc))

#L1, validation set y_pred
pred = y_pred_l1
true = y_val_test
fpr, tpr, thresh = metrics.roc_curve(true, pred)
auc = metrics.roc_auc_score(true, pred)
plt.plot(fpr,tpr,label="L1 pred (valid. set), auc="+str(auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.legend(loc=0)
plt.savefig('ROC_dr_v_models')
plt.show()


# This is the model used to run the Yoot Assess online assessment tool at www.yootassess.online
