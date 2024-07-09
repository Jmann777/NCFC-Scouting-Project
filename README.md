WORK IN PROGRESS


Introduction:
xG (expected goals) is a metric used within football to describe the probability of shot converting into a goal. The development of the xG metric has been used in statistical scouting by both clubs and consultancies when trying to identify potential players for their teams. The following project seeks to experimentally build 4 xG models in an attempt to identify potential scouting targets that Norwich City could have considered when they were required to replace Robbie Brady after his depature from the club. The 4 models were then applied and evaluated via logistic regression, random forest classification, and xGboost classification.

Data Collection:
The project combines event data across the top 5 European League (taken from statsbomb) with player valuations and minutes played (taken from transfermarkt) during the 2015/16 season. This data was chosen due to the larger sample size available across the 5 leagues with player valuations and mintues played supplying key features for player identification and normalisation. In an ideal context we would also include tracking data alongside the event data, however this is not available in open formats for the 2015/16 season.

Data Preparation:
- Event data was cleaned and prepped for model use. All shots were grouped and then separated into headers and all other shots. The logic behind this separation assumes that the conversion of a header will be different in most cases to other shots.
- Player valuation and minutes played data was matched using fuzzylogic + manual mapping to the event data.

Metric Selections for the Model:
The idea behind the use of 4 differing model was to separate shooting actions and the factors that affect goal conversion into four categories. These categories are:
- Basic (Incl - Angle, Distance, Inverse, League)
- Singular Player Effects (Incl - Basic + technique, first time shot)
- Teammate Effects (Incl - Player Effects + assist type, pattern of play)
- Opposition Effects (Incl - Teammate + under pressure, shot deflected)

Model Selection and Development:
The methods selected to model xG through this data were logistical regression, random forest classification, and xGboost classification. 
Add detailed reasoning for these

Model Evaluation:
Provide visualisations and stats for model scores. Potentially provide table or only the important models. Also add  reasoning for using statsbomb xg for visualisations.


What did Brady's data profile look like?
![Robbie Brady Radar](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/30b804c7-32a1-4094-91d4-9e11f7227292)


Which players look similar/better?

![Scatter plot for xG vs value](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/7e8e07f4-82c2-43f9-a16b-fa52286a75f2)

![xA Graph NCFC](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/267afb49-0073-4424-8d32-f4562af098c2)

![xGA Graph NCFC](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/55dc02b7-f15c-4bc4-b18c-8bcf1860ffc2)

![Filip Kostic Radar](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/9721d8e1-55ff-41ec-bf1e-9bda3f764a97)

![Jairo Samperio Radar](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/f66a3cc7-4ca3-4800-8ffe-a2f583d469de)

![Nicolas Benezet Radar](https://github.com/Jmann777/NCFC-Scouting-Project/assets/87671742/52f6f6bf-96ee-4d1f-8394-fef6187ded08)


Conclusion + Challenges
Top 3 targets and why?
Challenges + Improvements:
What challenges did I face? How could I improve the model? 
