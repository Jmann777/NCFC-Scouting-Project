"""
The following file engineers the features required for the model. This includes the allocation of the dependent variable
(goals) as well as the creation of independent variables which include distance and angle from the goal,
 pattern of play, finish technique, finish type, assist type, and pressure on shot.

This file also includes the creation of the 5 shot types that we will examine in our model (line 141-159).
"""


#todo Change to include all shots into the function
# - May need to delete the pickle references

import pickle

import pandas as pd
import numpy as np

# Data Import
with open('../Source/all_shots.pkl', 'rb') as file:
    all_shots = pickle.load(file)

with open('../Source/all_events.pkl', 'rb') as file:
    all_events = pickle.load(file)

with open('../Source/prem_shots.pkl', 'rb') as file:
    prem_shots = pickle.load(file)

prem_events = all_events[all_events["League"] == 1]
passes = prem_events[prem_events["type_name"] == "Pass"]
passes['key_passes'] = passes['pass_shot_assist'].notna().astype(int)
key_passes = passes[passes['pass_shot_assist'] == 1]


def model_metric_setup(shots: pd.DataFrame, shot_assists: pd.DataFrame):
    """
       Creation of model metrics. These metrics include distance, angle, pattern of play, finish technique,
       pressure on finish, shot off dribble, first time shot, and assist type.

       Parameters:
       - shots (pd.DataFrame): Dataframe containing all shots
       - shot_assists (pd.DataFrame): Dataframe containing all key passes that led to a shot on goal

       Returns:
       - combined_shots_passes (pd.Dataframe): Dataframe containing all shots and passes with the inclusion
       of the model metrics created within the function.
       """

    # Get the dependent variable (goals)
    shots["goal"] = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    shots["goal_smf"] = shots['goal'].astype(object)

    # Calculating angle and distance
    shots['x_ball'] = shots.x
    shots["x"] = shots.x.apply(lambda cell: 105 - cell)
    shots["c"] = shots.y.apply(lambda cell: abs(34 - cell))
    shots["angle"] = np.where(np.arctan(7.32 * shots["x"] / (
            shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)) >= 0, np.arctan(
        7.32 * shots["x"] / (shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)), np.arctan(
        7.32 * shots["x"] / (shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)) + np.pi) * 180 / np.pi
    shots["distance"] = np.sqrt(shots["x"] ** 2 + shots["c"] ** 2)

    # Prepping pressure column for the model (allocating nan = 0)
    shots["under_pressure"] = shots["under_pressure"].fillna(0)

    # Creating a dummy for pattern of play - Throw in, Corners, Free Kick, Regular
    shots["pattern_of_play"] = shots.play_pattern_name.apply(
        lambda cell: 1 if cell == 'From Throw In' else
        2 if cell == 'From Corner' else
        3 if cell == 'From Free Kick' else
        4 if cell == 'Regular Play' else 0)

    # Creating a dummy for finish technique - Normal, Header, Volley, Half Volley, Lob, Backheel, overhead kick
    shots["technique"] = shots.technique_name.apply(
        lambda cell: 1 if cell == 'Normal' else
        3 if cell == 'Volley' else
        4 if cell == 'Half Volley' else
        5 if cell == 'Lob' else
        6 if cell == 'Backheel' else
        7 if cell == "Overhead Kick" else 0)
    shots["technique"] = np.where(shots["body_part_name"] == "Head", 2, shots["technique"])

    # Converting shots off dribble into a dummy of 0,1
    shots["shot_follows_dribble"] = shots["shot_follows_dribble"].fillna(False).astype(int)

    # Converting first time finishes into dummy of 0,1
    shots["shot_first_time"] = shots["shot_first_time"].fillna(False).astype(int)

    # Creating a dummy for assist type - requires loop
    def assist_type(row):
        # Allocating headed assist type - headed regular (1), headed through ball (2), headed cutback (3)
        if row['body_part_name'] == "Head":
            if row['pass_cut_back'] == True:
                return 1
            elif row['technique_name'] == "Through Ball":
                return 2
            else:
                return 3
        # Allocating cross assist type - ground cross (4), low cross (5), regular cross (6)
        elif row['pass_cross'] == True:
            if row['pass_height_name'] == "Ground Pass":
                return 4
            elif row['pass_height_name'] == "Low Pass":
                return 5
            else:
                return 6
        # Allocating through ball assist type - ground through ball (7), low through ball (8), high throw ball (9)
        elif row['technique_name'] == "Through Ball":
            if row['pass_height_name'] == "Ground Pass":
                return 7
            elif row['pass_height_name'] == "Low Pass":
                return 8
            else:
                return 9
        # Allocating cutback (10)
        elif row['pass_cut_back'] == True:  # todo does this overwrite the headed cutback?
            return 10
        # Allocating regular assists (Ground(11), low(12), high(13))
        else:
            if row['pass_height_name'] == "Ground Pass":
                return 11
            elif row['pass_height_name'] == "Low Pass":
                return 12
            else:
                return 13

    shot_assists['assist_type'] = shot_assists.apply(assist_type, axis=1)
    shots = shots.merge(shot_assists[['id', 'assist_type']], left_on='shot_key_pass_id', right_on='id', how='left')

    combined_shots_passes = pd.concat([shots, shot_assists])
    combined_shots_passes = combined_shots_passes.sort_index()

    return combined_shots_passes


shot_passes: pd.DataFrame = model_metric_setup(prem_shots, key_passes)
shots_df = shot_passes[shot_passes["type_name"] == "Shot"]

# Removing penalties for xnpG
shots_df = shots_df[shots_df.sub_type_name != "Penalty"]

# Separating shots into subsections for modelling
shots_from_fk: pd.DataFrame = shots_df[shots_df["sub_type_name"] == "Free Kick"]

headed_shots_from_crosses: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] == "Head")
    & (shots_df["assist_type"].isin([4, 5, 6]))]

regular_shots_from_crosses: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] != "Head")
    & (shots_df["assist_type"].isin([4, 5, 6]))]

headed_shots_not_from_crosses: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] == "Head")
    & (~shots_df["assist_type"].isin([4, 5, 6]))]

regular_shots: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] != "Head")
    & (~shots_df["assist_type"].isin([4, 5, 6]))
    & (shots_df["sub_type_name"] != "Free Kick")]
