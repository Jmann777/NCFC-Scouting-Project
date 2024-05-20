import statsbomb_jm as sbj

pl_events = sbj.events_season(2,27)

naismith_events = pl_events.loc[pl_events["player_name"] == "Steven Naismith"]
print(naismith_events.size)

l