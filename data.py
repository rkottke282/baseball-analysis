import pandas as pd
import numpy as np
import os.path as path
helpful_columns = ['inning','outs','p_score','b_score','stand', \
                        'top', 'b_count', 's_count', 'pitch_num', \
                        'on_1b', 'on_2b', 'on_3b', \
                        'f_count', 'o_count', \
                        'prev_pitch_class','pitch_class']
fastballs = ['FA', 'FC', 'FF', 'FS', 'FT', 'SI']
offspeeds = ['CH', 'CU', 'EP', 'KC', 'KN', 'SC', 'SL']
nonhittables = ['AB', 'FO', 'PO', 'UN', 'IN']

# pass in no parameters to get all atbats.
# pass in a pitchers id to get just the rows corresponding to those at bats
def load_atbats(pitcher_id = ''):
    atbats_15to18 = pd.read_csv('./raw_data/atbats.csv')
    atbats_15to18 = atbats_15to18.reindex(sorted(atbats_15to18), axis=1)
    atbats_19 = pd.read_csv('./raw_data/2019_atbats.csv')
    atbats_19 = atbats_19.reindex(sorted(atbats_19), axis=1)
    atbats = pd.concat((atbats_15to18,atbats_19)) # 925634, 11
    if (pitcher_id != ''):
        atbats = atbats.loc[(atbats['pitcher_id'] == pitcher_id)]
    atbats['stand'] = atbats['stand'].replace('L', 1.0)
    atbats['stand'] = atbats['stand'].replace('R', 0.0)
    return atbats

# return pitch_class given pitch_type abbreviation code
def pitch_class(pitch_type):
    if (pitch_type in fastballs):
        return 0.0
    elif (pitch_type in offspeeds):
        return 1.0
    else:
        return -1.0

def append_previous_pitch_class(pitches):
    temp_pitches = pitches.copy().reset_index()
    temp_pitches['key'] = temp_pitches.apply(lambda x: int('{}{}'.format(int(x['ab_id']), int(x['pitch_num']))), axis=1)
    temp_pitches['prev_key'] = temp_pitches['key'].apply(lambda x: x-1)
    joined_temp_pitches = temp_pitches.merge(right=temp_pitches,left_on='prev_key', right_on='key', how='left')
    joined_temp_pitches = joined_temp_pitches[['index_x','pitch_class_y']]
    pitches = pitches.merge(right=joined_temp_pitches, left_index=True, right_on='index_x')
    pitches = pitches.rename(columns={'pitch_class_y':'prev_pitch_class'})
    pitches['prev_pitch_class'] = pitches['prev_pitch_class'].replace(np.nan, -1.0)
    return pitches

def populate_strike_ball_atbat_counts(pitches):
    # add new columns for filling in
    pitches['f_count'] = 0
    pitches['o_count'] = 0
    # reminder: 0 is a fastball, 1 is an offspeed
    fastballs_in_count = 0 
    offspeeds_in_count = 0
    current_ab_id = -1
    for idx, pitch in pitches.iterrows():
        # Reset the counts when the at bat changes
        if (current_ab_id != pitch.ab_id):
            current_ab_id = pitch.ab_id
            fastballs_in_count = 0
            offspeeds_in_count = 0

        # keep count of pitch types in at bat
        if (pitch.pitch_class == 0.0):
            fastballs_in_count += 1
        elif(pitch.pitch_class == 1.0):
            offspeeds_in_count += 1
        pitches.loc[idx,'f_count'] = fastballs_in_count 
        pitches.loc[idx,'o_count'] = offspeeds_in_count
    return pitches

# pass in no parameters to get all pitches.
# pass in a list of at bat ids to get just the rows corresponding to those at bats
def load_pitches(atbat_ids = ''):
    pitches_15to18 = pd.read_csv('./raw_data/pitches.csv')
    pitches_15to18 = pitches_15to18.reindex(sorted(pitches_15to18), axis=1)
    pitches_19 = pd.read_csv('./raw_data/2019_pitches.csv')
    pitches_19 = pitches_19.reindex(sorted(pitches_19), axis=1)
    pitches = pd.concat((pitches_15to18, pitches_19)) # 3595944, 40
    if (atbat_ids != ''):
        pitches = pitches.loc[(pitches['ab_id'].isin(atbat_ids))]

    # based on pitch_type, build a pitch_class field to distinguish fastballs from off-speed pitches    
    pitches['pitch_class'] = pitches['pitch_type'].apply(lambda x: pitch_class(x))

    # remove anything that's not a fastball or off-speed pitch (ie. Pitch out, Intentional Ball, etc.)
    pitches = pitches[pitches['pitch_class'] != -1.0]

    pitches = append_previous_pitch_class(pitches)
    pitches = populate_strike_ball_atbat_counts(pitches)

    return pitches

# pass in no parameters to get all players.
# pass in a firstname and lastname to get just the row corresponding to that player
def load_players(first_name = '', last_name = ''):
    players = pd.read_csv('./raw_data/player_names.csv') # 2218, 3
    if ((first_name != '') & (last_name != '')):
        players = players.loc[(players['first_name'] == first_name) & (players['last_name'] == last_name)]
    return players

# load all the data
def load_baseball_data():
    return load_atbats(), load_pitches(), load_players()

# get the strasbug data
# pass True to just get a subset of the data
def get_strasburg_data(subset=False):
    #get the player_id for Stephen Strasburg
    ss_id = int(load_players('Stephen', 'Strasburg').id)

    #get all of the at bats pitched by Stephen Strasburg's id
    ss_atbats = load_atbats(ss_id)
    if (subset):
        ss_atbats = ss_atbats.head(100)
    ss_atbat_ids = [int(x) for x in ss_atbats.ab_id]

    #get all of the pitches thrown by at bats in which Stephen Strasburg is pitching
    ss_pitches = load_pitches(ss_atbat_ids)

    #merge atbats and pitch datasets
    ss_atbats_pitches = ss_atbats.merge(ss_pitches, left_on='ab_id', right_on='ab_id')
    
    return ss_atbats_pitches
    
# cut the number of columns down, default uses helpful_columns defined at top
def reduce_columns(data, columns_to_keep=helpful_columns):
    return data[helpful_columns] 

# get only the data that meets the count criteria
def get_data_with_count(data, balls, strikes):
    data = data.loc[(data['b_count'] == balls) & (data['s_count'] == strikes)]
    return data

#Testing
# ss_data = reduce_columns(get_strasburg_data(False))
# ss_data = pd.read_csv('ss_data_test.csv')
# ss_data_reduced = get_data_with_count(ss_data, 0, 0)
# print(ss_data_reduced)
# ss_data.to_csv('ss_data_test.csv')

