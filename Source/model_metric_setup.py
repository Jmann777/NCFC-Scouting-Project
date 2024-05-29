import pickle

# Data Import
# pl_events = sbj.events_season(2, 27)
with open('../all_shots.pkl', 'rb') as file:
    prem_events = pickle.load(file)

# pl_shots = sbj.shots_season(2, 27)

with open('../prem_shots.pkl', 'rb') as file:
    prem_shots = pickle.load(file)

#

# Shot type (Dummy / split into DFs)- #todo plan is to apply each group of metrics to each shot type - Header, Free kick, Shot off carry, Shot off assists

# Distance
# Angle
l
# Type of assist - Ground pass, low pass, high pass, crosses, through balls, cutbacks? (Dummy)

# Pressure (Change nan to 0)

# Pattern of play - Throw in, corners, regular (Dummy)

# Technique - Normal, Volley, Header

# League (Dummy)