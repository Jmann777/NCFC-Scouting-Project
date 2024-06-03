"""
The following file imports the 2015/16 Big 5 Leagues open Statsbomb data and creates two shot dataframes.
These shot dataframes are saved into a pickle (for computing speed) and are later used for the xG model.
(Data info available at: https://statsbomb.com/news/the-2015-16-big-5-leagues-free-data-release-premier-league/)
"""

from Statsbomb_Data_Manipulations import statsbomb_jm as sbj
import pickle
import pandas as pd

# Data import via dictionary
cid = {'ENG': 2, 'SPN': 11, 'DE': 9, 'IT': 12, 'FR': 7}
sid = 27

competition_df = {}
for competition, comp_id in cid.items():
    print(f"Fetching data for {competition}...")
    competition_df[competition] = sbj.events_season(comp_id, sid)

prem_events = competition_df['ENG']
liga_events = competition_df['SPN']
bund_events = competition_df['DE']
serie_events = competition_df['IT']
ligue_events = competition_df['FR']

# Assigning league dummy for future use in the xG model
prem_events["League"] = 1
liga_events["League"] = 2
bund_events["League"] = 3
serie_events["League"] = 4
ligue_events["League"] = 5

# Creating a shots dataframe for all leagues and prem only for future use (DF saved to pickle)
all_events: pd.DataFrame = pd.concat([prem_events, liga_events, bund_events, serie_events, ligue_events])
all_events.reset_index(drop=True, inplace=True)
with open('all_events.pkl', 'wb') as file:
    pickle.dump(all_events, file)

all_shots: pd.DataFrame = all_events.loc[all_events["type_name"] == "Shot"]
with open('all_shots.pkl', 'wb') as file:
    pickle.dump(all_shots, file)

prem_shots: pd.DataFrame = prem_events.loc[prem_events["type_name"] == "Shot"]
with open('prem_shots.pkl', 'wb') as file:
    pickle.dump(prem_shots, file)