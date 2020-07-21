import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from collections import defaultdict

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data'   to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(False))
    ss_data.to_csv('ss_data.csv')

headers = ss_data.columns
#columns: 'inning', 'outs', 'p_score', 'b_score', 'stand', 'top', 'b_count','s_count', 'pitch_num', 'on_1b', 'on_2b', 'on_3b', 'prev_pitch_class','pitch_class'

np_ss_data = ss_data.to_numpy()
predictors = np_ss_data[:,:-1]
response = np_ss_data[:,-1].astype(int)

# Let's take a look at the coefficients of the parameters
lr_model = LogisticRegression().fit(predictors, response)
coefs = lr_model.coef_[0]
# for name, value in zip(headers, coefs):
#     print('{} has coefficient: {}'.format(name, value))
plt.bar([x for x in headers[:-1]], coefs)
plt.show()

# Now let's see what parameters are suggested using auc as a deciding factor
lr_model2 = LogisticRegression().fit(predictors, response)
rfecv = RFECV(lr_model2, scoring='roc_auc').fit(predictors, response)
print('Optimal number of features for logistic regression fitting: {}'.format(rfecv.n_features_))
rank_by_headers = defaultdict(str)
for idx,x in enumerate(rfecv.ranking_):
    existing = rank_by_headers.get(x)
    if (existing == None):
        rank_by_headers.update({x: headers[idx]})
    else:
        rank_by_headers.update({x: headers[idx] + ',' + existing})

print('Parameters by descending order of importance using auc as the scorer:')
for x in sorted(rank_by_headers.items()):
    print(x)

# Because we care most about the accuracy of the predictor, let's see what parameters are suggested using accuracy as a deciding factor
lr_model3 = LogisticRegression().fit(predictors, response)
rfecv = RFECV(lr_model3, scoring='accuracy').fit(predictors, response)
print('Optimal number of features for logistic regression fitting: {}'.format(rfecv.n_features_))
rank_by_headers = defaultdict(str)
for idx,x in enumerate(rfecv.ranking_):
    existing = rank_by_headers.get(x)
    if (existing == None):
        rank_by_headers.update({x: headers[idx]})
    else:
        rank_by_headers.update({x: headers[idx] + ',' + existing})

print('Parameters by descending order of importance using accuracy as the scorer:')
for x in sorted(rank_by_headers.items()):
    print(x)

#Using CART to determine feature importance
dtc = DecisionTreeClassifier().fit(predictors,response)
plt.bar([x for x in headers[:-1]], dtc.feature_importances_)
plt.show()


rank_by_headers = defaultdict(str)
for idx,x in enumerate(dtc.feature_importances_):
    existing = rank_by_headers.get(x)
    if (existing == None):
        rank_by_headers.update({x: headers[idx]})
    else:
        rank_by_headers.update({x: headers[idx] + ',' + existing})

print('Parameters by descending order for decision tree:')
for x in sorted(rank_by_headers.items()):
    print(x)

#Using Random Forest to determine feature importance
rfc = RandomForestClassifier().fit(predictors,response)
plt.bar([x for x in headers[:-1]], rfc.feature_importances_)
plt.show()

rank_by_headers = defaultdict(str)
for idx,x in enumerate(dtc.feature_importances_):
    existing = rank_by_headers.get(x)
    if (existing == None):
        rank_by_headers.update({x: headers[idx]})
    else:
        rank_by_headers.update({x: headers[idx] + ',' + existing})

print('Parameters by descending order for random forest:')
for x in sorted(rank_by_headers.items()):
    print(x)