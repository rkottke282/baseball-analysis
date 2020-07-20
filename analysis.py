import data as dataHelper
import pandas as pd
import numpy as np
import os.path as path


if (path.exists('ss_data.csv')):
    ss_data = pd.read_csv('ss_data.csv', index_col=0)
else:
    # Boolean in 'get_strasburg_data' to determine if we should get all the data (false)
    #  or just a subset (true)
    ss_data = dataHelper.reduce_columns(dataHelper.get_strasburg_data(True))
    ss_data.to_csv('ss_data.csv')

# Number of pitches
total_pitch_count = len(ss_data)
print('Total number of pitches thrown by Stephen Strasburg: {}'.format(total_pitch_count))

# Breakdown of pitch classes
pitch_classes = set(ss_data['pitch_class'].values)
pitch_class_summaries = []
for pitch_class in pitch_classes:
    num_class = len(ss_data[ss_data['pitch_class'] == pitch_class])
    percent_class = round(100 * num_class / total_pitch_count,2)
    pitch_class_summaries.append((pitch_class, \
        num_class, \
        '{}%'.format(percent_class)))

# Fastballs: 6874 -> 54.43%, Off-speeds: 5755 -> 46.57%
print(pitch_class_summaries) 

