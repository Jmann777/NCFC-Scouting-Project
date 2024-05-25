""" The following file xxx"""

import statsbomb_jm as sbj
import pickle

# Data import via dictionary
# cid = {'ENG':2, 'SPN':11, 'DE':9, 'IT':12, 'FR':7}
cid = 2
sid = 27
#prem = sbj.events_season(cid, sid)

#with open('prem.pkl', 'wb') as file:
#    pickle.dump(prem, file)

with open('prem.pkl', 'rb') as file:
    prem_loaded = pickle.load(file)

# competition_df = {}
# for competition, comp_id in cid.items():
#    print(f"Fetching data for {competition}...")
#    competition_df[competition] = sbj.events_season(comp_id, sid)

# prem_events = competition_df['ENG']
# Commented out for computing speed
# liga_events = competition_df['SPN']
# bund_events = competition_df['DE']
# serie_events = competition_df['IT']
# ligue_events = competition_df['FR']
# todo recomment in when finished with code (cid)

import pandas as pd


def get_minutes_played(events):
    match_ids = events['match_id'].unique()

    def get_minutes_played_single_match(match_select):
        per_match = events[events['match_id'] == match_select]
        match_length = per_match['minute'].max()

        lineups = sbj.lineups(match_select)
        lineups = lineups[['player_name', 'team_name']].copy()
        lineups['time_start'] = 0

        substitutions = per_match[per_match['type_name'] == "Substitution"]
        substitutions_out = substitutions[['minute', 'team_name', 'player_name']].copy()
        substitutions_out.rename(columns={'minute': 'time_out'}, inplace=True)

        line_ups = pd.merge(lineups, substitutions_out, on='player_name', how='left')

        substitutions_in = pd.DataFrame({
            'player_name': substitutions['substitution_replacement_name'],
            'team_name': substitutions['team_name'],
            'time_start': substitutions['minute'],
            'minute': match_length
        })

        player_ids = per_match.groupby('player_name')['player_id'].unique()
        player_ids = player_ids[player_ids.index.isin(substitutions_in['player_name'])]  # was player

        substitutions_in = pd.merge(substitutions_in, player_ids, left_on='player_name', right_index=True)

        line_ups = pd.concat([line_ups, substitutions_in])
        line_ups.fillna(match_length, inplace=True)

        line_ups['time_out'] = line_ups[['time_out', 'minute']].min(axis=1)
        line_ups['minutes_played'] = line_ups['time_out'] - line_ups['time_start']
        line_ups = line_ups.drop(columns=['minute'])

        line_ups['match_id'] = match_select
        line_ups = line_ups.rename(columns={
            'player': 'player_name',
            'player_id': 'player_id',
            'team_name': 'team_name',
            'time_start': 'time_in',
            'minute': 'time_out',
            'minutes_played': 'minutes_played',
            'match_id': 'match_id'
        })

        return line_ups

    mins_played_list = [get_minutes_played_single_match(match_id) for match_id in match_ids]
    mins_played_df = pd.concat(mins_played_list)

    # Ensure we keep only the second instance of each match_id for each player
    mins_played_df_sorted = mins_played_df.sort_values(by=["player_name", "match_id", "minutes_played"],
                                                       ascending=[True, True, False])
    mins_played_df_unique = mins_played_df_sorted.drop_duplicates(subset=['player_name', 'match_id'], keep='last')

    # Filter for Steven Naismith
    Naismith_mins = mins_played_df_unique[mins_played_df_unique["player_name"] == "Steven Naismith"]
    mins_played_df = mins_played_df_unique.groupby('player_name')['minutes_played'].sum().reset_index()

    #todo Currently player minutes aren't correlating to the correct players. Either this is completely wrong, or the merging/sum has not worked. Investigate why.

    return mins_played_df, Naismith_mins

minutes_played, Naismith = get_minutes_played(prem_loaded)




l
