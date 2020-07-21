import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

np_ss_data = ss_data.to_numpy()
predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=.2, random_state=282)

#Logistic regression with all predictors
lr_model = LogisticRegression().fit(X_train, y_train)
y_test_predictions = lr_model.predict(X_test)
print('Accuracy of logistic regression {}%'.format(round(100*accuracy_score(y_test, y_test_predictions),2)))
# Accuracy of log reg: 57.96%

#Per learning in variable-selection: variables by importance -> pitch_num, b_count, on_1b, stand, outs, s_count
# secondary variables -> on_3b, inning, prev_pitch_class
# tertiary variables -> top, b_score, on_2b, p_score

# Log Regression with top 9 predictors
np_ss_data = ss_data[['pitch_num', 'b_count', 'on_1b','stand','outs','s_count','on_3b', 'inning', 'prev_pitch_class','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)
lr_model = LogisticRegression().fit(X_train, y_train)
y_test_predictions = lr_model.predict(X_test)
print('Accuracy of logistic regression with 9 predictors {}%'.format(round(100*accuracy_score(y_test, y_test_predictions),2)))
# Accuracy of log reg: 58.79%

# Log Regression with top 6 predictors
np_ss_data = ss_data[['pitch_num', 'b_count', 'on_1b','stand','outs','s_count','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)
lr_model = LogisticRegression().fit(X_train, y_train)
y_test_predictions = lr_model.predict(X_test)
print('Accuracy of logistic regression with 6 predictors {}%'.format(round(100*accuracy_score(y_test, y_test_predictions),2)))
# Accuracy of log reg: 57.96%

# Log Regression with top 2 predictors
np_ss_data = ss_data[['pitch_num', 'b_count','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)
lr_model = LogisticRegression().fit(X_train, y_train)
y_test_predictions = lr_model.predict(X_test)
print('Accuracy of logistic regression with 2 predictors {}%'.format(round(100*accuracy_score(y_test, y_test_predictions),2)))
# Accuracy of log reg: 57.96%

#Random Forest
rfc = RFC(max_depth=6).fit(X_train, y_train)
rfc_y_test_predictions = rfc.predict(X_test)
print('Accuracy of random forest regression {}%'.format(round(100*accuracy_score(y_test, rfc_y_test_predictions),2)))

#Neural Network
for i in [3, 10, 25, 75, 100]:
    nn = MLPClassifier(hidden_layer_sizes=i).fit(X_train, y_train)
    nn_y_test_predictions = nn.predict(X_test)
    print('Accuracy of neural network with {} hidden layer(s): {}%'.format(i, round(100*accuracy_score(y_test, nn_y_test_predictions),2)))

#Naive Bayes
nbc = GaussianNB().fit(X_train, y_train)
nbc_y_test_predictions = nbc.predict(X_test)
print('Accuracy of Naive Bayes {}%'.format(round(100*accuracy_score(y_test, nbc_y_test_predictions),2)))

#Naive Bayes with priors
nbc = GaussianNB(priors=[.5443,.4557]).fit(X_train, y_train)
nbc_y_test_predictions = nbc.predict(X_test)
print('Accuracy of Naive Bayes with priors {}%'.format(round(100*accuracy_score(y_test, nbc_y_test_predictions),2)))
