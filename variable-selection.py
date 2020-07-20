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

np_ss_data = ss_data.to_numpy()
predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=.2, random_state=282)

# Let's do forward variable selection, adding variables one at a time and keeping variables which 
#  contribute to a better model, using AUC as a measure of accuracy
lr_model = LogisticRegression().fit(X_train, y_train)
y_test_predictions = lr_model.predict(X_test)



