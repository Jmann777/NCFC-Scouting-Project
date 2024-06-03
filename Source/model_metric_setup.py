import pickle

import pandas as pd
import numpy as np

#todo consider break up pof assist type - do we need to group into one column? or can i create a column fore each type breakdown e.g. column 1 = crosses, column 2 = headers etc

# Data Import
# pl_events = sbj.events_season(2, 27)
with open('../Source/all_shots.pkl', 'rb') as file:
    all_shots = pickle.load(file)

with open('../Source/all_events.pkl', 'rb') as file:
    all_events = pickle.load(file)
# pl_shots = sbj.shots_season(2, 27)

with open('../Source/prem_shots.pkl', 'rb') as file:
    prem_shots = pickle.load(file)

prem_events = all_events[all_events["League"] == 1]
passes = prem_events[prem_events["type_name"] == "Pass"]
passes['key_passes'] = passes['pass_shot_assist'].notna().astype(int)
key_passes = passes[passes['pass_shot_assist'] == 1]
#key_passes = passes.pass_shot_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
#passes["key_passes"] = key_passes
#key_passes = passes[passes["pass_shot_assist"] == 1]


def model_metric_setup(shots: pd.DataFrame, shot_assists: pd.DataFrame):
    """
       Obtain goals xxx #todo this

       Parameters:
       - shots (pd.DataFrame): Dataframe containing all shots

       Returns:
       - shots (pd.Dataframe): Dataframe containing model variables including goals(DV) location, angle, distance(IVs)
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
                     7 if cell == "Overhead kick" else 0)
    shots["technique"] = np.where(shots["body_part_name"] == "Head", 2, shots["technique"])


    # Creating a dummy for assist type- requires function
    def assist_type(row):
        # Allocating headed assist type - headed regular (1), headed through ball (2), headed cutback (3)
        if row['body_part_name'] == "Head":
            if row['pass_cut_back'] == "True":
                return 1
            elif row['technique_name'] == "Through Ball":
                return 2
            else:
                return 3
        # Allocating cross assist type - ground cross (4), low cross (5), regular cross (6)
        elif row['pass_cross'] == "True":
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
        elif row['pass_cut_back'] == "True": #todo does this overwrite the headed cutback?
            return 10
        # Allocating regular assists (11)
        else:
            return 11

    shot_assists['assist_type'] = shot_assists.apply(assist_type, axis=1)
    shots = shots.merge(shot_assists[['id', 'assist_type']], left_on='shot_key_pass_id', right_on='id', how='left')


    combined_shots_passes = pd.concat([shots, shot_assists])
    combined_shots_passes = combined_shots_passes.sort_index()

    return combined_shots_passes

test = model_metric_setup(prem_shots, key_passes)

headers = key_passes[key_passes["assist_type"].isin([1, 2, 3])]
crosses = key_passes[key_passes["assist_type"].isin([4, 5, 6])]
through_balls = key_passes[key_passes["assist_type"].isin([7, 8, 9])]
cutbacks = key_passes[key_passes["assist_type"] == 10]
other = key_passes[key_passes["assist_type"] == 11]

#todo
# Fix the cutback and crosses True logic
# shots only
# test that everything works



l



# Shot type (Dummy / split into DFs)- see sub_type_name ot help

#todo plan is to apply each group of metrics to each shot type - Header, Free kick, Shot off carry, Shot off assists
