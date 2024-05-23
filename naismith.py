import statsbomb_jm as sbj
import pandas as pd
import numpy as np
import matplotlib as plt
from mplsoccer import Pitch

# Data Import
pl_events = sbj.events_season(2, 27)
pl_shots = sbj.shots_season(2, 27)

# Data setup - Filter down to player events + shots per team
naismith_events = pl_events.loc[pl_events["player_name"] == "Steven Naismith"].reset_index()
naismith_events.to_pickle(naismith_events)
naismith_ncfc = naismith_events.loc[naismith_events["team_name"] == "Norwich City"].reset_index()
naismith_ncfc.to_pickel(naismith_ncfc)
naismith_everton = naismith_events.loc[naismith_events["team_name"] == "Everton"].reset_index()
naismith_everton.to_pickel(naismith_everton)
naismith_shots = pl_shots.loc[pl_shots["player_name"] == "Steven Naismith"].reset_index()
naismith_shots.to_pickle(naismith_shots)

# Minutes played
#for i, match_id in pl_events():
    # Order events
#    pl_events.sort_values(by="Minute") #todo this may sort all by minute rather than match id?
    # Define time events e.g. Start of halves, end of halves, position change, substitution
    # "Period" = half, ["type_name"] == "Half Start/End", ["type_name"] == "Tactical Shift", ["type_name"] == "Substitution"


# naismith_shots_ncfc = naismith_shots[naismith_shots["team_name"] == "Norwich City"].reset_index()
# naismith_shots_everton = naismith_shots[naismith_shots["team_name"] == "Everton"].reset_index()

# todo do I want to add ball receipts in final third?
# todo calculate player minutes with event data or find dataset
# todo sort merge and drop specifications
# todo do I use len(df) to count instead of .sum?

# Role + playing style
# Average position
def average_pass_lcoation(events):
    passes = events[events["type_name"] == "Pass"]
    scatter_df = pd.DataFrame()
    for i, name in enumerate(passes["player_name"].unique()):
        passx = passes.loc[passes["player_name"] == name]["x"].to_numpy()
        recx = passx.loc[passes["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = passes.loc[passes["player_name"] == name]["y"].to_numpy()
        recy = passy.loc[passes["pass_recipient_name"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player_name"] = name
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))

        # Drawing pitch
        pitch = Pitch(line_color='grey')
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                             endnote_height=0.04, title_space=0, endnote_space=0)
        # Scatter the location on the pitch
        pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1,
                      alpha=1, ax=ax["pitch"], zorder=3)
        # annotating player name
        for i, row in scatter_df.iterrows():
            pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold",
                           size=16, ax=ax["pitch"], zorder=4)

        fig.suptitle("Naismith average pass and reception location", fontsize=30)
        plt.show()

average_pass_lcoation(naismith_events)





# Offensive Goal - npGoals, Shots, npxG, shot selection (xG/shots), finishing (Goals-xG) #todo remember to pass shot logic
def shot_statistics(shots) -> pd.DataFrame:
    """ Obtains and adjusts the number shots, non-penalty goals, npxG as well as assessing shot selection (xG/shots)
    and quality of finishing (non-penalty goals - npxG)

    Returns:
    - shot_stats (pd.DataFrame): Dataframe containing shot related data for Naismith. Specifically, shots, npG, npxG,
    shot selections and quality of finishing
    """
    # Shots
    shots_player: pd.DataFrame = shots.size().reset_index(name='shots')
    shot_per90: pd.DataFrame = shots_player.merge(m_played, on="player_name")  # todo find minutes played data
    shot_per90.loc[shot_per90['Player_Minutes'] > 400, "shot_p90"] = (
            (shot_per90["shots"] / shot_per90['Player_Minutes']) * 90)

    # Goals
    goals: pd.Series = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    shots["non_pen_goals"] = goals  # todo might need to consider non_pen goals
    npg_player: pd.DataFrame = shots[["player_id", "player_name", "non_pen_goals"]]
    # npg_player = npg_player.groupby(["player_id", "player_name"])["non_pen_goals"].sum().reset_index()
    npg_per90: pd.DataFrame = npg_player.merge(m_played, on="player_name")
    npg_per90.loc[npg_per90['Player_Minutes'] > 400, "npg_p90"] = (
            (npg_per90["non_pen_goals"] / npg_per90['Player_Minutes']) * 90)

    # xG
    xg_player: pd.DataFrame = shots["shot_statsbomb_xg"].sum()
    xg_per90: pd.DataFrame = xg_player.merge(m_played, on="player_name")
    xg_per90.loc[xg_per90['Player_Minutes'] > 400, "xg_p90"] = (
            (xg_per90["shot_statsbomb_xg"] / xg_per90['Player_Minutes']) * 90)
    xg_per90 = xg_per90.sort_values(by='player_id').reset_index(drop=True)

    # Shot selection and quality
    shot_selection = xg_per90["xg_p90"] / shot_per90["shot_p90"]
    finish_quality = npg_per90["npg_p90"] - xg_per90["xg_p90"]

    shot_stats: pd.DataFrame = pd.concat([shot_per90,
                                          npg_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          xg_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          shot_selection.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          finish_quality.drop(columns=['player_name', 'Player_Minutes', 'player_id'])],
                                         axis=1)

    return shot_stats


# Offensive playmaking
def pass_statistics(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """ Obtains and adjusts the number assists, progressive passes, key passes, key pass xG, and pass completion rate
    Key passes are defined as passes that resulted in a shot

    Parameters:
    - shots (pd.DataFrame): Dataframe containing shot data from statsbomb
    - events_xg (pd.DataFrame): Dataframe containing all events

    Returns:
    - pass_data (pd.DataFrame): Dataframe containing pass related data for all players. Specifically, number of assists,
    key passes, progressive passes, key pass xG, and pass completion rate
    """
    # Assists
    passes = events.loc[events["type_name"] == "Pass"]
    assists: pd.Series = passes.pass_goal_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['assists'] = assists
    assist_player: pd.DataFrame = passes[["player_id", "player_name", "assists"]]
    # assist_player = assist_player.groupby(["player_id", "player_name"])["assists"].sum().reset_index()
    assist_per90: pd.DataFrame = assist_player.merge(m_played, on="player_name")
    assist_per90.loc[assist_per90['Player_Minutes'] > 400, "assists_p90"] = (
            (assist_per90["assists"] / assist_per90['Player_Minutes']) * 90)

    # Key passes
    key_passes: pd.Series = passes.pass_shot_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['key_passes'] = key_passes
    key_passes_player: pd.DataFrame = passes[["player_id", "player_name", "key_passes"]]
    # key_passes_player = key_passes_player.groupby(["player_id", "player_name"])["key_passes"].sum().reset_index()
    key_passes_per90: pd.DataFrame = key_passes_player.merge(m_played, on="player_name")
    key_passes_per90.loc[key_passes_per90['Player_Minutes'] > 400, "key_pass_p90"] = (
            (key_passes_per90["key_passes"] / key_passes_per90['Player_Minutes']) * 90)

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
    key_pass_xg_player: pd.DataFrame = key_pass_events.groupby("player_name")["shots_statsbomb_xg"].sum().reset_index()
    key_pass_xg_player.rename(columns={"our_xg": "key_pass_xg"}, inplace=True)
    key_pass_xg_player = key_pass_xg_player.merge(m_played, on="player_name")
    key_pass_xg_player.loc[key_pass_xg_player['Player_Minutes'] > 400, "key_pass_xg_p90"] = (
            (key_pass_xg_player["key_pass_xg"] / key_pass_xg_player['Player_Minutes']) * 90)

    # Pass completion rate
    successful = passes.loc[passes["outcome_name"] == "nan"]
    unsuccessful = passes.loc[passes["outcome_name"] == "Incomplete"]
    completion_rate = successful / unsuccessful * 100

    # Progressive passes
    successful["start"] = np.sqrt(np.square(120 - successful["x"]) + np.square(40 - successful["y"]))
    successful["end"] = np.sqrt(np.square(120 - successful["end_x"]) + np.square(40 - successful["end_y"]))
    successful["progressive"] = [(successful['end'][x]) / (successful["start"][x])
                                 < .75 for x in range(len(successful.start))]
    prog_passes = successful["progressive"].reset_index()
    prog_passes.loc[prog_passes['Player_Minutes'] > 400, "key_pass_xg_p90"] = (
            (key_pass_xg_player["key_pass_xg"] / key_pass_xg_player['Player_Minutes']) * 90)


    pass_stats: pd.DataFrame = pd.concat([assist_per90,
                                          key_passes_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          key_pass_xg_player.drop(columns=['player_name', 'Player_Minutes']),
                                          prog_passes,
                                          completion_rate], axis=1)

    return pass_stats


def dribble_statistics(events):
    """ Obtains number of progressive carries, 1v1s, and success rates

    Parameters:
    - events (pd.DataFrame): Dataframe containing all events

    Returns:
    - dribble_data (pd.DataFrame): Dataframe containing dribble related data for all players. Specifically, progressive carries,
    final third carries, and 1v1s.

    """
    carries = events[(events["type_name"] == "Carry")]
    carries.merge(m_played, on="player_name")
    # Progressive carries
    carries["start"] = np.sqrt(np.square(120 - carries["x"]) + np.square(40 - carries["y"]))
    carries["end"] = np.sqrt(np.square(120 - carries["end_x"]) + np.square(40 - carries["end_y"]))
    carries["progressive"] = [(carries['end'][x]) / (carries["start"][x]) < .90 for x in range(len(carries.start))]
    prog_carries = carries["progressive"]
    prog_carries.loc[prog_carries['Player_Minutes'] > 400, "prog_carries_p90"] = (
            (prog_carries.sum() / prog_carries['Player_Minutes']) * 90)

    # 1v1
    take_ons = events[(events["type_name"] == "Dribble")]
    take_ons.merge(m_played, on="player_name")
    successful_take_ons = take_ons[(take_ons["outcome_name"] == "Complete")]
    successful_take_ons.loc[successful_take_ons['Player_Minutes'] > 400, "succ_take_ons_p90"] = (
            (successful_take_ons.sum() / successful_take_ons['Player_Minutes']) * 90)
    take_on_rate = successful_take_ons.sum() - take_ons.sum()

    dribble_stats: pd.DataFrame = pd.concat([prog_carries,
                                             successful_take_ons.drop(
                                                 columns=['player_name', 'Player_Minutes', 'player_id']),
                                             take_on_rate.drop(columns=['player_name']),
                                             ], axis=1)

    return dribble_stats


# Defensive -
def defensive_statistics(events):
    """ xxx """
    # Pressures
    pressures = events[events["type_name"] == "Pressure"]
    # Pressures in the final third
    final_third_pressures = pressures[pressures["x"] > 80]
    num_final_third_pressures = final_third_pressures.sum()
    # Ball recoveries in final third
    recoveries = events[events["type_name"] == "Ball Recovery"]
    recoveries = recoveries[recoveries["outcome_name"] == "Successful"] #todo check to ensure this is correct
    recoveries_final_third = recoveries[recoveries["x"] > 80]

    def_stats = pressures.merge(recoveries,
                                recoveries_final_third,
                                num_final_third_pressures)
    return def_stats