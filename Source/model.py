"""
The following file ...
"""
# Importing packages
import pickle
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve

# Opening pickle files for each shot type
with open('../Source/fk.pkl', 'rb') as file:
    free_kicks = pickle.load(file)

with open('../Source/headers.pkl', 'rb') as file:
    headers = pickle.load(file)

with open('../Source/regular_shots.pkl', 'rb') as file:
    regular_shots = pickle.load(file)

# Examining shape for test/train splits
print("Free kick", free_kicks.shape)
print("Headers", headers.shape)
print("Regular", regular_shots.shape)

# Setting independent variables for each model
basic_x = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League']
opposition_x_fk = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League', 'shot_deflected']

player_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                  'technique', 'shot_first_time']
teammate_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                    'technique', 'shot_first_time',
                    'pattern_of_play', 'assist_type']
opposition_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                      'technique', 'shot_first_time',
                      'pattern_of_play', 'assist_type',
                      'under_pressure', 'shot_deflected']


# todo change to "league"
def splits_by_league(data, test_size=0.2):
    """

    :param df:
    :param test_size:
    :return:
    """
    data = data.sort_values(by=['League', 'match_id'])

    train_frames = []
    test_frames = []

    # Get unique leagues
    leagues = data['League'].unique()

    for league in leagues:
        league_data = data[data['League'] == league]

        # Split the data
        train_data, test_data = train_test_split(league_data, test_size=test_size, shuffle=False)

        # Append to respective lists
        test_frames.append(test_data)
        train_frames.append(train_data)

    # Concatenate all test and train frames
    train_data = pd.concat(train_frames)
    test_data = pd.concat(test_frames)

    return train_data, test_data


def logistic_model(data, x):
    """
    This function...
    :param data:
    :return:
    """
    # Splitting the data
    train_data, test_data = splits_by_league(data, test_size=0.2)

    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    x_vars = x
    y_var = 'goal_smf'

    x_train = train_data[x_vars]
    y_train = train_data[y_var]
    x_test = train_data[x_vars]
    y_test = train_data[y_var]

    # Creating dummies for league
    x_train['League'] = x_train['League'].astype('category')
    x_test['League'] = x_test['League'].astype('category')
    x_train = pd.get_dummies(x_train, columns=['League'], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=['League'], drop_first=True)

    # Ensure x_test has the same columns as x_train
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    # Running logistic regression
    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Model Evaluation
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")

    logloss = log_loss(y_test, y_pred)
    print(f"Log-Loss: {logloss}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    brier = brier_score_loss(y_test, y_pred)
    print(f"Brier Score: {brier}")

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print(f"AUC: {auc}")

    return y_pred


#fk_model_1 = logistic_model(free_kicks, basic_x)
#fk_model_2 = logistic_model(free_kicks, opposition_x_fk)
#header_model1 = logistic_model(headers, basic_x)
#header_model2 = logistic_model(headers, player_x_other)
#header_model3 = logistic_model(headers, teammate_x_other)
#header_model4 = logistic_model(headers, opposition_x_other)
regular_shots_model1 = logistic_model(regular_shots, basic_x)
regular_shots_model2 = logistic_model(regular_shots, player_x_other)
regular_shots_model3 = logistic_model(regular_shots, teammate_x_other)
regular_shots_model4 = logistic_model(regular_shots, opposition_x_other)

#todo y_pred vs y_pred_prob
# todo how can I evaluate this model?
# - Test for multicollinearity - vif
# - Create other logit models
# - Investigate whether it is easier to set categorcial varibales in the metric engineering


# Compare my xg to statsbomb

# https://www.youtube.com/watch?v=zM4VZR0px8E
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
