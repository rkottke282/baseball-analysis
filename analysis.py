import data as dataHelper
import pandas as pd
import os.path as path

if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(True))
    ss_data.to_csv('ss_data.csv')

print(set(ss_data['pitch_class'].values))
