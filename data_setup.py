""" The following file xxx"""

import statsbomb_jm as sbj

# Data import via dictionary
#cid = {'ENG':2, 'SPN':11, 'DE':9, 'IT':12, 'FR':7}
cid = 2
sid = 27

competition_df = {}
for competition, comp_id in cid.items():
    print(f"Fetching data for {competition}...")
    competition_df[competition] = sbj.events_season(comp_id, sid)

prem_events = competition_df['ENG']
# Commented out for computing speed
#liga_events = competition_df['SPN']
#bund_events = competition_df['DE']
#serie_events = competition_df['IT']
#ligue_events = competition_df['FR']
#todo recomment in when finished with code (cid)
l

