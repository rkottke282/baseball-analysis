import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

np_ss_data = ss_data[['prev_pitch_class','o_count','f_count','pitch_num','b_count','pitch_class']].to_numpy()
red_predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(red_predictors, response, test_size=.2, random_state=282)

nn = MLPClassifier(hidden_layer_sizes=3).fit(X_train, y_train)
nn_y_test_predictions = nn.predict(X_test)
print('Accuracy of neural network with 3 hidden layer(s): {}%'.format(round(100*accuracy_score(y_test, nn_y_test_predictions),2)))
print(nn.coefs_)
print(nn.intercepts_)