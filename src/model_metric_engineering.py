"""
The following file engineers the features required for the model. This includes the allocation of the dependent variable
(goals) as well as the creation of independent variables which include distance and angle from the goal,
 pattern of play, finish technique, finish type, assist type, and pressure on shot.
"""

import pickle

import pandas as pd
import numpy as np

# Data Import
with open('../data_logos/all_shots.pkl', 'rb') as file:
    all_shots = pickle.load(file)

with open('../data_logos/all_events.pkl', 'rb') as file:
    all_events = pickle.load(file)

passes = all_events[all_events["type_name"] == "Pass"]
passes['key_passes'] = passes['pass_shot_assist'].notna().astype(int)
key_passes = passes[passes['pass_shot_assist'] == 1]


def model_metric_setup(shots: pd.DataFrame, shot_assists: pd.DataFrame):
    """
       Creation of model metrics for logistic regression. These metrics include distance, angle, pressure on finish,
        shot off dribble, first time shot, and assist type. Note - pattern of play, finish technique, assists, and league
        are all categorised as dummies using one hot encoding in the model itself (see model.random_forest_model)

       Parameters:
       - shots (pd.DataFrame): Dataframe containing all shots
       - shot_assists (pd.DataFrame): Dataframe containing all key passes that led to a shot on goal

       Returns:
       - combined_shots_passes (pd.Dataframe): Dataframe containing all shots and passes with the inclusion
       of the model metrics created within the function.
       """

    # Get the dependent variable (goals)
    shots["goal"] = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    shots["goal_smf"] = shots['goal'].astype(int)

    # Calculating angle and distance + inverses
    shots['x_ball'] = shots.x
    shots["x"] = shots.x.apply(lambda cell: 105 - cell)
    shots["c"] = shots.y.apply(lambda cell: abs(34 - cell))
    shots["angle"] = np.where(np.arctan(7.32 * shots["x"] / (
            shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)) >= 0, np.arctan(
        7.32 * shots["x"] / (shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)), np.arctan(
        7.32 * shots["x"] / (shots["x"] ** 2 + shots["c"] ** 2 - (7.32 / 2) ** 2)) + np.pi) * 180 / np.pi
    shots["distance"] = np.sqrt(shots["x"] ** 2 + shots["c"] ** 2)

    epsilon = 1e-6
    shots["inverse_angle"] = 1 / (shots["angle"] + epsilon)

    shots["inverse_distance"] = 1 / (shots["distance"] + epsilon)

    # Prepping pressure column for the model (allocating nan = 0)
    shots["under_pressure"] = shots["under_pressure"].fillna(0)

    # Converting shots off dribble into a dummy of 0,1
    shots["shot_follows_dribble"] = shots["shot_follows_dribble"].fillna(False).astype(int)

    # Converting first time finishes into dummy of 0,1
    shots["shot_first_time"] = shots["shot_first_time"].fillna(False).astype(int)

    # Converting deflected shots into dummy of 0,1
    shots["shot_deflected"] = shots["shot_deflected"].fillna(False).astype(int)

    # Creating assist type - requires loop
    def assist_type(row):
        # Allocating headed assist type - headed cutback (1), headed through ball (2), regular (3)
        if row['body_part_name'] == "Head":
            if row['pass_cut_back']:
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
        # Allocating through ball assist type - ground through ball (7), low through ball (8), high through ball (9)
        elif row['technique_name'] == "Through Ball":
            if row['pass_height_name'] == "Ground Pass":
                return 7
            elif row['pass_height_name'] == "Low Pass":
                return 8
            else:
                return 9
        # Allocating cutback (10)
        elif row['pass_cut_back']:
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

    combined_shots_passes_trees = pd.concat([shots, shot_assists])
    combined_shots_passes_trees = combined_shots_passes_trees.sort_index()

    combined_shots_passes_trees['assist_type'] = combined_shots_passes_trees['assist_type'].fillna(False).astype(int)

    return combined_shots_passes_trees

# Combining shots and passes then filtering to shots only #todo check if i need to do this
shot_passes: pd.DataFrame = model_metric_setup(all_shots, key_passes)
shots_df = shot_passes[shot_passes["type_name"] == "Shot"]
shots_df = shots_df[shots_df.sub_type_name != "Penalty"]

# Separating shots into headers and non-headers
headers: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] == "Head")]
with open('../data_logos/headers.pkl', 'wb') as file:
    pickle.dump(headers, file)

regular_shots: pd.DataFrame = shots_df[
    (shots_df["body_part_name"] != "Head")]
with open('../data_logos/regular_shots.pkl', 'wb') as file:
    pickle.dump(regular_shots, file)
