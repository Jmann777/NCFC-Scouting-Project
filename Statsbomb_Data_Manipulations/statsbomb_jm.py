""" The following file imports match, event, and tracking data from statsbomb. It also creates a dataframe consisting
of shots from the imported event data
"""

import pandas as pd
from statsbombpy import sb
from mplsoccer import Sbopen


def matches(cid: int, sid: int) -> pd.DataFrame:
    """ Obtain seasonal match data.

    Parameters:
    - cid (int): Integer based on competition ID required to load statsbomb data
    - sid (int): Integer based on season ID required to load statsbomb data

    Returns:
    - matches (pd.Dataframe): Dataframe containing match information from statsbomb
    """
    parser = Sbopen()
    df_match: pd.DataFrame = parser.match(competition_id=cid, season_id=sid)
    df_matches: pd.DataFrame = df_match.match_id.unique()
    return df_matches

def lineups(match_id):
    parser = Sbopen()
    lineup: pd.DataFrame = parser.lineup(match_id)

    return lineup

def shots_season(cid: int, sid: int) -> pd.DataFrame:
    """ Obtain season shots data.

       Parameters:
    - cid (int): Integer based on competition ID required to load statsbomb data
    - sid (int): Integer based on season ID required to load statsbomb data

    Returns:
    - shot_df (pd.Dataframe): Dataframe containing all shot data from season
    """
    parser = Sbopen()
    df_match: pd.DataFrame = parser.match(competition_id=cid, season_id=sid)
    df_matches: pd.Series = df_match.match_id.unique()
    shot_df: pd.DataFrame = pd.DataFrame()
    for match in df_matches:
        parser = Sbopen()
        df_event: pd.DataFrame = parser.event(match)[0]
        shots: pd.DataFrame = df_event.loc[df_event["type_name"] == "Shot"]
        shots.x = shots.x.apply(lambda cell: cell * 105 / 120)
        shots.y = shots.y.apply(lambda cell: cell * 68 / 80)
        shot_df: pd.DataFrame = pd.concat([shot_df, shots], ignore_index=True)
    shot_df.reset_index(drop=True, inplace=True)
    return shot_df


def tracking_season(cid: int, sid: int) -> pd.DataFrame:
    """ Obtain seasonal tracking data.

       Parameters:
    - cid (int): Integer based on competition ID required to load statsbomb data
    - sid (int): Integer based on season ID required to load statsbomb data

    Returns:
    - track_df (pd.Dataframe): Dataframe containing tracking information from statsbomb 360 data
    """
    parser = Sbopen()
    df_match: pd.DataFrame = parser.match(competition_id=cid, season_id=sid)
    df_matches: pd.Series = df_match.match_id.unique()
    track_df: pd.DataFrame = pd.DataFrame()
    for match in df_matches:
        parser = Sbopen()
        df_track: pd.DataFrame = parser.event(match)[2]
        df_track.x = df_track.x.apply(lambda cell: cell * 105 / 120)
        df_track.y = df_track.y.apply(lambda cell: cell * 68 / 80)
        track_df: pd.DataFrame = pd.concat([track_df, df_track], ignore_index=True)
    track_df.reset_index(drop=True, inplace=True)
    return track_df


def events_season(cid: int, sid: int) -> pd.DataFrame:
    """ Obtain seasonal event data.

       Parameters:
    - cid (int): Integer based on competition ID required to load statsbomb data
    - sid (int): Integer based on season ID required to load statsbomb data

    Returns:
    - event_df (pd.Dataframe): Dataframe containing event information from the season
    """
    parser = Sbopen()
    df_match: pd.DataFrame = parser.match(competition_id=cid, season_id=sid)
    df_matches: pd.Series = df_match.match_id.unique()
    event_df: pd.DataFrame = pd.DataFrame()
    for match in df_matches:
        parser = Sbopen()
        df_event: pd.DataFrame = parser.event(match)[0]
        df_event.x = df_event.x.apply(lambda cell: cell * 105 / 120)
        df_event.y = df_event.y.apply(lambda cell: cell * 68 / 80)
        event_df: pd.DataFrame = pd.concat([event_df, df_event], ignore_index=True)
    event_df.reset_index(drop=True, inplace=True)
    return event_df

def events_single(match: int):
    df_match = sb.events(match)

    return df_match

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
        line_ups = line_ups.drop(columns=['minute'])

        players_with_events = per_match['player_name'].unique()
        line_ups.loc[~line_ups['player_name'].isin(players_with_events), 'time_out'] = 0

        line_ups['minutes_played'] = line_ups['time_out'] - line_ups['time_start']

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

    mins_played_df = mins_played_df_unique.groupby('player_name')['minutes_played'].sum().reset_index()

    return mins_played_df
#todo review + make not about min difference