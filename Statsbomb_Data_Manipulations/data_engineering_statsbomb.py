""" The following file is set up to allow for easy access to data engineering that is required to statsbomb data.
It is broken into the following sections: X Y Z"""

import pandas as pd
import numpy as np

# todo clean all of the functions. They are unbelievable inefficient

# Section 2 - Player statistics
def shot_statistics(shots: pd.DataFrame,
                    m_played: pd.DataFrame) -> pd.DataFrame:  # todo consider non_penalty goals as currently contains penalties
    """ Obtains and adjusts the number shots, non-penalty goals, npxG as well as assessing shot selection (xG/shots)
    and quality of finishing (non-penalty goals - npxG)

    Parameters:
    - shots (pd.DataFrame): Dataframe containing shot data from statsbomb
    - m_played (pd.DataFrame): Dataframe containing all minutes played per player

    Returns:
    - shot_stats (pd.DataFrame): Dataframe containing shot related data for Naismith. Specifically, shots, npG, npxG,
    shot selections and quality of finishing
    """
    # Shots
    shots_player: pd.DataFrame = shots.groupby("player_name").size().reset_index(name='shots')
    shot_per90: pd.DataFrame = shots_player.merge(m_played, on="player_name")
    shot_per90.loc[shot_per90['minutes_played'] > 400, "shot_p90"] = (
            (shot_per90["shots"] / shot_per90['minutes_played']) * 90)

    # Goals
    goals: pd.Series = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    shots["non_pen_goals"] = goals
    npg_player: pd.DataFrame = shots[["player_name", "non_pen_goals"]]
    npg_player = npg_player.groupby(["player_name"])["non_pen_goals"].sum().reset_index()
    npg_per90: pd.DataFrame = npg_player.merge(m_played, on="player_name")
    npg_per90.loc[npg_per90['minutes_played'] > 400, "npg_p90"] = (
            (npg_per90["non_pen_goals"] / npg_per90['minutes_played']) * 90)

    # xG
    xg_player: pd.DataFrame = shots.groupby('player_name')["shot_statsbomb_xg"].sum().reset_index()
    xg_per90: pd.DataFrame = xg_player.merge(m_played, on="player_name")
    xg_per90.loc[xg_per90['minutes_played'] > 400, "xg_p90"] = (
            (xg_per90["shot_statsbomb_xg"] / xg_per90['minutes_played']) * 90)
    xg_per90 = xg_per90.sort_values(by='player_name').reset_index(drop=True)

    # Shot selection and quality
    xg_per90["shot_selection"] = xg_per90["xg_p90"] / shot_per90["shot_p90"]
    xg_per90["finish_quality"] = npg_per90["npg_p90"] - xg_per90["xg_p90"]

    shot_per90.set_index('player_name', inplace=True)
    npg_per90.set_index('player_name', inplace=True)
    xg_per90.set_index('player_name', inplace=True)

    shot_stats: pd.DataFrame = pd.concat([shot_per90,
                                          npg_per90.drop(columns=['minutes_played']),
                                          xg_per90.drop(columns=['minutes_played']),
                                          ], axis=1)

    return shot_stats


def pass_statistics(events: pd.DataFrame, shots: pd.DataFrame, m_played: pd.DataFrame) -> pd.DataFrame:
    """ Obtains and adjusts the number assists, progressive passes, key passes, key pass xG, and pass completion rate
    Key passes are defined as passes that resulted in a shot

    Parameters:
    - shots (pd.DataFrame): Dataframe containing shot data from statsbomb
    - events_xg (pd.DataFrame): Dataframe containing all events
    - m_played (pd.DataFrame): Dataframe containing all minutes played per player

    Returns:
    - pass_data (pd.DataFrame): Dataframe containing pass related data for all players. Specifically, number of assists,
    key passes, progressive passes, key pass xG, and pass completion rate
    """
    # Assists
    passes = events.loc[events["type_name"] == "Pass"].reset_index()
    assists: pd.Series = passes.pass_goal_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['assists'] = assists
    assist_player: pd.DataFrame = passes[["player_name", "assists"]]
    assist_player = assist_player.groupby(["player_name"])["assists"].sum().reset_index()
    assist_per90: pd.DataFrame = assist_player.merge(m_played, on="player_name")
    assist_per90.loc[assist_per90['minutes_played'] > 400, "assists_p90"] = (
            (assist_per90["assists"] / assist_per90['minutes_played']) * 90)

    # Key passes
    key_passes: pd.Series = passes.pass_shot_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['key_passes'] = key_passes
    key_passes_player: pd.DataFrame = passes[["player_name", "key_passes"]]
    key_passes_player = key_passes_player.groupby(["player_name"])["key_passes"].sum().reset_index()
    key_passes_per90: pd.DataFrame = key_passes_player.merge(m_played, on="player_name")
    key_passes_per90.loc[key_passes_per90['minutes_played'] > 400, "key_pass_p90"] = (
            (key_passes_per90["key_passes"] / key_passes_per90['minutes_played']) * 90)

    # Key pass xG
    shots['key_passes'] = key_passes
    key_pass_events: pd.DataFrame = shots.loc[
        (shots["type_name"].isin(["Pass", "Shot"])) & (
                (shots["key_passes"] == 1) | (shots["type_name"] == "Shot"))]
    key_pass_events = key_pass_events.sort_values(by="possession")
    num_possession: int = max(key_pass_events["possession"].unique())
    for i in range(num_possession + 1):
        possession_chain: pd.DataFrame = key_pass_events.loc[key_pass_events["possession"] == i].sort_values(by="index")
        if len(possession_chain) > 0:
            if possession_chain.iloc[-1]["type_name"] == "Shot":
                xg: float = possession_chain.iloc[-1]["shot_statsbomb_xg"]
                key_pass_events.loc[key_pass_events["possession"] == i, 'shot_statsbomb_xg'] = xg
    key_pass_events = key_pass_events.loc[key_pass_events["key_passes"] == 1]
    key_pass_xg_player: pd.DataFrame = key_pass_events.groupby("player_name")["shot_statsbomb_xg"].sum().reset_index()
    key_pass_xg_player.rename(columns={"shot_statsbomb_xg": "key_pass_xg"}, inplace=True)
    key_pass_xg_player = key_pass_xg_player.merge(m_played, on="player_name")
    key_pass_xg_player.loc[key_pass_xg_player['minutes_played'] > 400, "key_pass_xg_p90"] = (
            (key_pass_xg_player["key_pass_xg"] / key_pass_xg_player['minutes_played']) * 90)

    # Progressive passes
    passes["start"] = np.sqrt(np.square(120 - passes["x"]) + np.square(40 - passes["y"]))
    passes["end"] = np.sqrt(np.square(120 - passes["end_x"]) + np.square(40 - passes["end_y"]))

    passes["progressive"] = passes.apply(lambda row: row["start"] < row["end"] - 10, axis=1)

    prog_passes = passes.loc[passes["progressive"] == True].groupby('player_name').size().reset_index(
        name='prog_passes')
    prog_passes = prog_passes.merge(m_played, on="player_name")
    prog_passes.loc[prog_passes['minutes_played'] > 400, "prog_passes_p90"] = (
            (prog_passes["prog_passes"] / prog_passes['minutes_played']) * 90)

    assist_per90.set_index('player_name', inplace=True)
    key_passes_per90.set_index('player_name', inplace=True)
    key_pass_xg_player.set_index('player_name', inplace=True)
    prog_passes.set_index('player_name', inplace=True)

    pass_stats: pd.DataFrame = pd.concat([assist_per90,
                                          key_passes_per90.drop(columns=['minutes_played']),
                                          key_pass_xg_player.drop(columns=['minutes_played']),
                                          prog_passes.drop(columns=['minutes_played']),
                                          ], axis=1)

    return pass_stats

def delivery_statistics(events: pd.DataFrame, m_played: pd.DataFrame):
    """ Obtains and adjusts the number attempted crosses, successful crosses, chances created from crosses,
    assists from crosses.

        Parameters:
        - events_xg (pd.DataFrame): Dataframe containing all events
        - m_played (pd.DataFrame): Dataframe containing all minutes played per player

        Returns:
        - delivery_data (pd.DataFrame):
        """


def dribble_statistics(events: pd.DataFrame, m_played: pd.DataFrame):
    """ Obtains number of progressive carries, 1v1s, and success rates

    Parameters:
    - events (pd.DataFrame): Dataframe containing all events
    - m_played (pd.DataFrame): Dataframe containing all minutes played per player

    Returns:
    - dribble_data (pd.DataFrame): Dataframe containing dribble related data for all players. Specifically, progressive carries,
    final third carries, and 1v1s.

    """
    carries = events[(events["type_name"] == "Carry")]
    carries.merge(m_played, on="player_name")
    # Progressive carries
    carries["start"] = np.sqrt(np.square(120 - carries["x"]) + np.square(40 - carries["y"]))
    carries["end"] = np.sqrt(np.square(120 - carries["end_x"]) + np.square(40 - carries["end_y"]))

    carries["progressive"] = carries.apply(lambda row: row["start"] < row["end"] - 5, axis=1)

    prog_carries = carries.loc[carries["progressive"] == True].groupby('player_name').size().reset_index(
        name='prog_carries')
    prog_carries = prog_carries.merge(m_played, on="player_name")
    prog_carries.loc[prog_carries['minutes_played'] > 400, "prog_carries_p90"] = (
            (prog_carries["prog_carries"] / prog_carries['minutes_played']) * 90)

    # 1v1
    take_ons = events[(events["type_name"] == "Dribble")].groupby('player_name').size().reset_index(name="take_ons")
    take_ons = take_ons.merge(m_played, on="player_name")
    take_ons.loc[take_ons['minutes_played'] > 400, "take_ons_p90"] = (
            (take_ons["take_ons"] / take_ons['minutes_played']) * 90)

    dribbles = events[(events["type_name"] == "Dribble")]
    successful_take_ons = dribbles[(dribbles["outcome_name"] == "Complete")].groupby('player_name').size().reset_index(
        name="successful_take_ons")
    successful_take_ons = successful_take_ons.merge(m_played, on="player_name")
    successful_take_ons.loc[successful_take_ons['minutes_played'] > 400, "succ_take_ons_p90"] = (
            (successful_take_ons["successful_take_ons"] / successful_take_ons['minutes_played']) * 90)

    successful_take_ons['take_on_rate'] = successful_take_ons['succ_take_ons_p90'] / take_ons[
        'take_ons_p90'] * 100  # todo fix

    prog_carries.set_index("player_name", inplace=True)
    take_ons.set_index("player_name", inplace=True)
    successful_take_ons.set_index("player_name", inplace=True)

    dribble_stats: pd.DataFrame = pd.concat([prog_carries,
                                             take_ons.drop(columns=['minutes_played']),
                                             successful_take_ons.drop(
                                                 columns=['minutes_played'])
                                             ], axis=1)

    return dribble_stats


def defensive_statistics(events: pd.DataFrame, m_played: pd.DataFrame):
    """ xxx """
    # Pressures
    pressures_full = events[events["type_name"] == "Pressure"]

    pressures_full["ft"] = pressures_full.apply(lambda row: row["x"] > 80, axis=1)

    pressures = events[events["type_name"] == "Pressure"].groupby('player_name').size().reset_index(name="pressures")
    pressures = pressures.merge(m_played, on="player_name")
    pressures.loc[pressures['minutes_played'] > 400, "pressures_p90"] = (
            (pressures["pressures"] / pressures['minutes_played']) * 90)

    # Pressures in the final third
    pressures_ft = pressures_full.loc[pressures_full["ft"] == True].groupby('player_name').size().reset_index(
        name='pressures_ft')
    pressures_ft = pressures_ft.merge(m_played, on="player_name")
    pressures_ft.loc[pressures_ft['minutes_played'] > 400, "pressures_ft_p90"] = (
            (pressures_ft["pressures_ft"] / pressures_ft['minutes_played']) * 90)

    # Ball recoveries
    recoveries = events[events["type_name"] == "Ball Recovery"].groupby('player_name').size().reset_index(
        name="recoveries")
    recoveries = recoveries.merge(m_played, on="player_name")
    recoveries.loc[recoveries['minutes_played'] > 400, "recoveries_p90"] = (
            (recoveries["recoveries"] / recoveries['minutes_played']) * 90)

    # recoveries final third
    recoveries_full = events[events["type_name"] == "Ball Recovery"]
    recoveries_full["ft"] = recoveries_full.apply(lambda row: row["x"] > 80, axis=1)
    recoveries_ft = pressures_full.loc[pressures_full["ft"] == True].groupby('player_name').size().reset_index(
        name='recoveries_ft')
    recoveries_ft = recoveries_ft.merge(m_played, on="player_name")
    recoveries_ft.loc[recoveries_ft['minutes_played'] > 400, "recoveries_ft_p90"] = (
            (recoveries_ft["recoveries_ft"] / recoveries_ft['minutes_played']) * 90)

    pressures.set_index('player_name', inplace=True)
    pressures_ft.set_index('player_name', inplace=True)
    recoveries.set_index('player_name', inplace=True)
    recoveries_ft.set_index('player_name', inplace=True)

    def_stats: pd.DataFrame = pd.concat([pressures,
                                         pressures_ft.drop(columns=["minutes_played"]),
                                         recoveries.drop(columns=["minutes_played"]),
                                         recoveries_ft.drop(columns=["minutes_played"]),
                                         ], axis=1)

    return def_stats
