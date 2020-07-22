import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

# np_ss_data = ss_data.to_numpy()
# predictors = np_ss_data[:,:-1]
# response = np_ss_data[:,-1].astype(int)
# X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=.2, random_state=282)

# Logistic regression classifier based on pitch_num, s_count, b_count, f_count, o_count, prev_pitch_class
np_ss_data = ss_data[['pitch_num', 's_count', 'b_count', 'f_count', 'o_count','prev_pitch_class','pitch_class']].to_numpy()
predictors_1 = np_ss_data[:,:-1]
response_1 = np_ss_data[:,-1].astype(int)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(predictors_1, response_1, test_size=.2, random_state=282)
lr_model_1 = LogisticRegression().fit(X_train_1, y_train_1)
y_test_predictions_1 = lr_model_1.predict(X_test_1)
print('Accuracy of logistic regression round 1: {}%'.format(round(100*accuracy_score(y_test_1, y_test_predictions_1),2)))
