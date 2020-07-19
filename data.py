import pandas as pd
helpful_columns = ['inning','outs','p_score','b_score','stand', \
                        'top', 'b_count', 's_count', 'pitch_num', \
                        'on_1b', 'on_2b', 'on_3b', 'type', 'pitch_type']

# pass in no parameters to get all atbats.
# pass in a pitchers id to get just the rows corresponding to those at bats
def load_atbats(pitcher_id = ''):
    atbats_15to18 = pd.read_csv('./data/atbats.csv')
    atbats_15to18 = atbats_15to18.reindex(sorted(atbats_15to18), axis=1)
    atbats_19 = pd.read_csv('./data/2019_atbats.csv')
    atbats_19 = atbats_19.reindex(sorted(atbats_19), axis=1)
    atbats = pd.concat((atbats_15to18,atbats_19)) # 925634, 11
    if (pitcher_id != ''):
        atbats = atbats.loc[(atbats['pitcher_id'] == pitcher_id)]
    return atbats

# pass in no parameters to get all pitches.
# pass in a list of at bat ids to get just the rows corresponding to those at bats
def load_pitches(atbat_ids = ''):
    pitches_15to18 = pd.read_csv('./data/pitches.csv')
    pitches_15to18 = pitches_15to18.reindex(sorted(pitches_15to18), axis=1)
    pitches_19 = pd.read_csv('./data/2019_pitches.csv')
    pitches_19 = pitches_19.reindex(sorted(pitches_19), axis=1)
    pitches = pd.concat((pitches_15to18, pitches_19)) # 3595944, 40
    if (atbat_ids != ''):
        pitches = pitches.loc[(pitches['ab_id'].isin(atbat_ids))]
    return pitches

# pass in no parameters to get all players.
# pass in a firstname and lastname to get just the row corresponding to that player
def load_players(first_name = '', last_name = ''):
    players = pd.read_csv('./data/player_names.csv') # 2218, 3
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




