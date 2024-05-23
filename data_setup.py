""" The following file xxx"""

import statsbomb_jm as sbj

# Data import via dictionary
#cid = {'ENG':2, 'SPN':11, 'DE':9, 'IT':12, 'FR':7}
cid = 2
sid = 27
prem = sbj.events_season(cid, sid)

#competition_df = {}
#for competition, comp_id in cid.items():
#    print(f"Fetching data for {competition}...")
#    competition_df[competition] = sbj.events_season(comp_id, sid)

#prem_events = competition_df['ENG']
# Commented out for computing speed
#liga_events = competition_df['SPN']
#bund_events = competition_df['DE']
#serie_events = competition_df['IT']
#ligue_events = competition_df['FR']
#todo recomment in when finished with code (cid)

import pandas as pd


def get_minutes_played(events):
    match_ids = events['match_id'].unique()

    def get_minutes_played_single_match(match_select):
        per_match = events[events['match_id'] == match_select]
        match_length = per_match['minute'].max()

        line_up1 = per_match.iloc[0]['tactics']['lineup']
        for player in line_up1:
            player['team_name'] = per_match.iloc[0]['team']['name']

        line_up2 = per_match.iloc[1]['tactics']['lineup']
        for player in line_up2:
            player['team_name'] = per_match.iloc[1]['team']['name']

        line_ups = pd.DataFrame(line_up1 + line_up2)
        line_ups = line_ups[['player', 'team_name']].copy()
        line_ups['time_start'] = 0

        substitutions = per_match[per_match['type']['name'] == "Substitution"]
        substitutions_out = substitutions[['minute', 'team']['name', 'player']['name']].copy()

        line_ups = pd.merge(line_ups, substitutions_out, left_on='player', right_on='player', how='left')

        substitutions_in = pd.DataFrame({
            'player': substitutions['substitution']['replacement']['name'],
            'team_name': substitutions['team']['name'],
            'time_start': substitutions['minute'],
            'minute': match_length
        })

        player_ids = per_match.groupby('player')['player']['id'].unique()
        player_ids = player_ids[player_ids.index.isin(substitutions_in['player'])]

        substitutions_in = pd.merge(substitutions_in, player_ids, left_on='player', right_index=True)

        line_ups = pd.concat([line_ups, substitutions_in])
        line_ups.fillna(match_length, inplace=True)
        line_ups['minutes_played'] = line_ups['minute'] - line_ups['time_start']

        line_ups['match_id'] = match_select
        line_ups = line_ups.rename(columns={
            'player': 'player_name',
            'player.id': 'player_id',
            'team.name': 'team_name',
            'time_start': 'time_in',
            'minute': 'time_out',
            'minutes_played': 'minutes_played',
            'match_id': 'match_id'
        })

        return line_ups

    mins_played_list = [get_minutes_played_single_match(match_id) for match_id in match_ids]
    mins_played_df = pd.concat(mins_played_list)

    return mins_played_df


# Example usage:
# events = pd.read_json('path_to_statsbomb_event_data.json')
# minutes_played = get_minutes_played(events)
# print(minutes_played)

minutes_played = get_minutes_played(prem)


l