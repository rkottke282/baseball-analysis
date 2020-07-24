import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

# Log Regression - 5 features
five_predictor_ss_data = ss_data[['prev_pitch_class','o_count','f_count','pitch_num','b_count','s_count','pitch_class']]
five_pred = five_predictor_ss_data[['prev_pitch_class','o_count','f_count','pitch_num','s_count','b_count']]
five_resp = five_predictor_ss_data[['pitch_class']]
X_train, X_test, y_train, y_test = train_test_split(five_pred, five_resp, test_size=.2, random_state=282)
X_train = X_train[['prev_pitch_class','o_count','f_count','pitch_num','b_count']]
test_blank_counts = dataHelper.get_data_with_count(X_test, 0, 0)[['prev_pitch_class','o_count','f_count','pitch_num','b_count']]
y_test_blanks = y_test.join(test_blank_counts, how='right').pitch_class
test_nonblank_counts = dataHelper.get_data_with_count(X_test, 0, 0, True)[['prev_pitch_class','o_count','f_count','pitch_num','b_count']]
y_test_noblanks = y_test.join(test_nonblank_counts, how='right').pitch_class

# evaluate trained lr_5 on only 0-0 counts
lr_5 = LogisticRegression().fit(X_train, y_train.to_numpy().ravel())
blank_count_predictions_lr5 = lr_5.predict(test_blank_counts)
print('Accuracy of logistic regression with 5 predictors on blank count: {}%'.format(round(100*accuracy_score(y_test_blanks.to_numpy().ravel(), blank_count_predictions_lr5.ravel()),2)))

# evaluate trained lr_5 on all other counts
nonblank_count_predictions_lr5 = lr_5.predict(test_nonblank_counts)
print('Accuracy of logistic regression with 5 predictors on non blank counts: {}%'.format(round(100*accuracy_score(y_test_noblanks.to_numpy().ravel(), nonblank_count_predictions_lr5.ravel()),2)))

#Random Forest
rfc = RFC(max_depth=6).fit(X_train, y_train.to_numpy().ravel())
rfc_y_test_predictions_blanks = rfc.predict(test_blank_counts)
print('Accuracy of random forest on blank counts {}%'.format(round(100*accuracy_score(y_test_blanks, rfc_y_test_predictions_blanks.ravel()),2)))

#Random Forest
rfc_y_test_predictions_nonblanks = rfc.predict(test_nonblank_counts)
print('Accuracy of random forest on nonblank counts {}%'.format(round(100*accuracy_score(y_test_noblanks, rfc_y_test_predictions_nonblanks.ravel()),2)))

#Neural Network
nn = MLPClassifier(hidden_layer_sizes=100).fit(X_train, y_train.to_numpy().ravel())
nn_y_test_predictions_blanks = nn.predict(test_blank_counts)
print('Accuracy of neural network on blank count: {}%'.format(round(100*accuracy_score(y_test_blanks, nn_y_test_predictions_blanks.ravel()),2)))

#Neural Network
nn_y_test_predictions_noblanks = nn.predict(test_nonblank_counts)
print('Accuracy of neural network on non-blank count: {}%'.format(round(100*accuracy_score(y_test_noblanks, nn_y_test_predictions_noblanks.ravel()),2)))

