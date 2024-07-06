Intoduction:
Goal + Context
Brady = 5.2mil Euro market gain (52% increase in market value)

Data Collection:
What data is being used? How have I cleaned it?(Adjusted to 105/120- figure out what this is) How have I normalised it?(Per 90) Which features have I added? (Minutes played)

Role analysis:
Robbie Brady - Traditional winger - Lots of crosses, decent amount of shots (Data viz?), ball winning,  shot map?, pass map?

Do I want to include xGOT?

Metric Selection:
Why have I chosen the metrics?
- Distance
- Angle
- GK position? Need tracking data
- Shot type - Headers vs non_headers, directly from dribble, directly from free kick 
- Pattern of play - Set piece, counter attack, open play
- Type of assist - High cross, low cross, through ball, long ball, pass after through ball, cutbacks
- Watch worville

Model Dev:
What models have I sued and why? What was the output?
xG, xA, xA from crosses
Logistical reg
random forest
xG boost (maybe)

Model Evaluation:
Logit - with predicted proibabilities- 
Free kick m1- Log loss = 0.24; brier = 0.06; auc = 0.60
Free kick m2 - Log loss = 0.23; brier = 0.06; auc = 0.62

Headers m1- Log loss = 0.3; brier = 0.08; auc = 0.72
Headers m2- Log loss = 0.3; brier = 0.09; auc = 0.72
Headers m3- Log loss = 0.08; brier = 0.02, auc = 0.98
Headers m4- Log loss = 0.08; brier = 0.03; auc = 0.98

Regular m1- Log loss = 0.26; brier = 0.07; auc = 0.79
Regualr m2- Log loss = 0.26; brier = 0.07; auc = 0.80
Regular m3- Log loss = 0.16; brier = 0.05; auc = 0.95
Regular m4- Log loss = 0.16; brier = 0.05; auc = 0.95 (Best)


Which players look good?


Challenges + Improvements:
What challenges did I face? How could I improve the model? (Age profile, HG status)
Data generation? Supervised clustering of players to target psotion and player type

Conclusion:
Top 3 targets and why?


https://steveaq.github.io/Player-Roles-Clustering/
https://cartilagefreecaptain.sbnation.com/2015/10/19/9295905/premier-league-projections-and-new-expected-goals
xg model - https://github.com/Slothfulwave612/xG-Model

model setup:
fit - https://www.youtube.com/watch?v=K5sVKQw9QgU
Inter vs Extrap - https://www.youtube.com/watch?v=rY5pdNW7jKM
Think about which one? Probably inter if only applying to one league?