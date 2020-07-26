import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

# Log Regression with top 8 predictors
np_ss_data = ss_data[['prev_pitch_class','o_count','f_count','pitch_num','b_count','s_count','on_2b','on_3b','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)
lr_model_8 = LogisticRegression().fit(X_train, y_train)
lr_test_prob_8 = lr_model_8.predict_proba(X_test)[:,1]
lr_auc_8 = roc_auc_score(y_test, lr_test_prob_8)
print('AUC of linear regression classifier with 8 features: {}'.format(lr_auc_8))

# Log Regression with top 5 predictors
np_ss_data = ss_data[['prev_pitch_class','o_count','f_count','pitch_num','b_count','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)
lr_model = LogisticRegression().fit(X_train, y_train)
lr_test_prob = lr_model.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test, lr_test_prob)
print('AUC of linear regression classifier with 5 features: {}'.format(lr_auc))

#Random Forest
rfc = RFC(max_depth=6).fit(X_train, y_train)
rfc_test_prob = rfc.predict_proba(X_test)[:,1]
rfc_auc = roc_auc_score(y_test, rfc_test_prob)
print('AUC of random forest classifier: {}'.format(rfc_auc))

#Neural Network
nn = MLPClassifier().fit(X_train, y_train)
nn_test_prob = nn.predict_proba(X_test)[:,1]
nn_auc = roc_auc_score(y_test, nn_test_prob)
print('AUC of neural network: {}'.format(nn_auc))

#Plotting ROC AUC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_test_prob)
lr_fpr_8, lr_tpr_8, _ = roc_curve(y_test, lr_test_prob_8)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rfc_test_prob)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_test_prob)
plt.plot(lr_fpr, lr_tpr, color='Red', label='Logistic Regression w/ 5 Predictors')
plt.plot(lr_fpr_8, lr_tpr_8, color='Orange', label='Logistic Regression w/ 8 Predictors')
plt.plot(rf_fpr, rf_tpr, color='Blue', label='Random Forest')
plt.plot(nn_fpr, nn_tpr, color='Black', label='Neural Network')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend()
plt.title('ROC AUC for various classifiers')
plt.show()