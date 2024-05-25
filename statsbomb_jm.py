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