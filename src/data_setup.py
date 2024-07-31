"""
The following file imports the 2015/16 Big 5 Leagues open Statsbomb data and creates two shot dataframes.
These shot dataframes are saved into a pickle (for computing speed) and are later used for the xG model.
(Data info available at: https://statsbomb.com/news/the-2015-16-big-5-leagues-free-data-release-premier-league/)
"""

from statsbomb_data_manipulations import statsbomb_jm as sbj
import pickle
import pandas as pd

# Data import via dictionary
cid = {'ENG': 2, 'SPN': 11, 'DE': 9, 'IT': 12, 'FR': 7}
sid = 27

competition_df = {}
for competition, comp_id in cid.items():
    print(f"Fetching data for {competition}...")
    competition_df[competition] = sbj.events_season(comp_id, sid)

prem_events: pd.DataFrame = competition_df['ENG']
liga_events: pd.DataFrame = competition_df['SPN']
bund_events: pd.DataFrame = competition_df['DE']
serie_events: pd.DataFrame = competition_df['IT']
ligue_events: pd.DataFrame = competition_df['FR']

# Assigning league dummy for future use in the xG model
prem_events["league"] = 1
liga_events["league"] = 2
bund_events["league"] = 3
serie_events["league"] = 4
ligue_events["league"] = 5

# Creating an event dataframe, with match date included, for all leagues for future use (DF saved to pickle)
all_events: pd.DataFrame = pd.concat([prem_events, liga_events, bund_events, serie_events, ligue_events])

all_events.reset_index(drop=True, inplace=True)
with open('../data_logos/all_events.pkl', 'wb') as file:
    pickle.dump(all_events, file)

all_shots: pd.DataFrame = all_events.loc[all_events["type_name"] == "Shot"]
with open('../data_logos/all_shots.pkl', 'wb') as file:
    pickle.dump(all_shots, file)
