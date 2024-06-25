"""
The following file ...
"""
# Importing packages
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler


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
def splits_by_league(data, test_size=0.1):
    """
    Splits the data by league into training, validation, and test sets.

    Parameters:
       - data (pd.DataFrame): Dataframe containing input data for model and data splits
       - test_size (int): Percentage of overall dataframe to be split into the test data
       - validation_size (int): Percentage of overall dataframe to be split into the validation data

    Returns:
         - Training and test dataframes for model
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


def plot_learning_curve(estimator, X_train, y_train, cv=None, scoring='neg_log_loss'):
    """
    Plots a learning curve for the inputted model results

    Parameters:
    - estimator: The object that fits and predicts e.g. the model run
    - X_train (np.array): Training vector containing n_sample and n_features
    - y_train (np.array): Target for classification e.g. goals
    - cv (int): Number of cross validations. Set to none as default of 5 is used
    - scoring (str): Type of model scoring/evaluation

    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)
    )

    # Handling negating values from the use of "neg_log_loss"
    if 'neg_' in scoring:
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
    else:
        # Calculating mean and sd over scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

    # Plotting
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="r",
                     alpha=0.1)
    plt.plot(train_sizes, val_scores_mean, label="Validation score", color="g")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color="g",
                     alpha=0.1)
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score" if 'neg_' not in scoring else scoring)
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def logistic_model(data, x):
    """
    Fits a logistic regression model with L2 regularization and evaluates its performance.

    Parameters:
    - data (pd.DataFrame): Dataframe containing input data for model and data splits
    - x: The features used as IVs to predict Y

    Returns:
    - Predicted probabilities for the test data.
    """
    # Loading in and sorting the data
    train_data, test_data = splits_by_league(data, test_size=0.1)

    x_vars = x
    y_var = 'goal_smf'

    x_train = train_data[x_vars].copy()
    y_train = train_data[y_var]
    x_test = test_data[x_vars].copy()
    y_test = test_data[y_var]

    # Creating dummies
    x_train['League'] = x_train['League'].astype('category')
    x_test['League'] = x_test['League'].astype('category')
    x_train = pd.get_dummies(x_train, columns=['League'], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=['League'], drop_first=True)

    # Aligning sets to avoid data leakage
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    # Apply Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert scaled arrays back to DataFrames with the original column names
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    penalty = 'l2'
    C = 0.001

    model = LogisticRegression(penalty=penalty, C=C, max_iter=10000)
    model.fit(x_train_scaled, y_train)

    x_data_scaled = pd.concat([x_train_scaled, x_test_scaled])
    y_data = pd.concat([y_train, y_test])
    xg_values = model.predict_proba(x_data_scaled)[:, 1]

    cv_scores = cross_val_score(model, x_data_scaled, y_data, cv=5, scoring='roc_auc')

    print(f"Cross-Validation AUC Scores: {cv_scores}")
    print(f"Mean Cross-Validation AUC Score: {cv_scores.mean()}")

    y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]

    train_accuracy = model.score(x_train_scaled, y_train)
    test_accuracy = model.score(x_test_scaled, y_test)
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")

    logloss = log_loss(y_test, y_pred_prob)
    print(f"Log-Loss: {logloss}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    brier = brier_score_loss(y_test, y_pred_prob)
    print(f"Brier Score: {brier}")

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print(f"AUC: {auc}")

    plot_learning_curve(model, x_train_scaled, y_train, cv=5, scoring='neg_log_loss')

    return xg_values


def random_forest_model(data, x):
    """
    Fits a random forest model and evaluates its performance

     Parameters:
    - data (pd.Dataframe): Dataframe containing input data for model and data splits
    - x: The features used as IVs to predict Y

    Returns:
    - Predicted probabilities for the test data.
    """
    train_data, test_data = splits_by_league(data, test_size=0.1)
    x_vars = x
    y_var = 'goal_smf'

    x_train = train_data[x_vars]
    y_train = train_data[y_var]
    x_test = test_data[x_vars]
    y_test = test_data[y_var]

    # Creating dummies for league
    x_train['League'] = x_train['League'].astype('category')
    x_test['League'] = x_test['League'].astype('category')
    x_train = pd.get_dummies(x_train, columns=['League'], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=['League'], drop_first=True)

    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    # Apply Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert scaled arrays back to DataFrames with the original column names
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    # Running Forest
    classifier = RandomForestClassifier(n_estimators=200)
    classifier.fit(x_train, y_train)

    # Cross-Validation
    x_data_scaled = pd.concat([x_train_scaled, x_test_scaled])
    y_data = pd.concat([y_train, y_test])
    xg_values = classifier.predict_proba(x_data_scaled)[:, 1]

    cv_scores = cross_val_score(classifier, x_data_scaled, y_data, cv=5, scoring='roc_auc')

    y_pred_prob = classifier.predict_proba(x_test)[:, 1]

    # Model Evaluation
    train_accuracy = classifier.score(x_train_scaled, y_train)
    test_accuracy = classifier.score(x_test_scaled, y_test)
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")

    logloss = log_loss(y_test, y_pred_prob)
    print(f"Log-Loss: {logloss}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    brier = brier_score_loss(y_test, y_pred_prob)
    print(f"Brier Score: {brier}")

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print(f"AUC: {auc}")

    plot_learning_curve(classifier, x_train, y_train, cv=5, scoring='neg_log_loss')

    return xg_values2


# Free kick models
# fk_model_1 = logistic_model(free_kicks, basic_x)
# fk_model_2 = logistic_model(free_kicks, opposition_x_fk)
# fk_model_1_rf = random_forest_model(free_kicks, basic_x)
# fk_model_2_rf = random_forest_model(free_kicks, opposition_x_fk)

# Header models
# header_model1 = logistic_model(headers, basic_x)
# header_model2 = logistic_model(headers, player_x_other)
# header_model3 = logistic_model(headers, teammate_x_other)
# header_model4 = logistic_model(headers, opposition_x_other)
# header_model1_rf = random_forest_model(headers, basic_x)
# header_model2_rf = random_forest_model(headers, player_x_other)
# header_model3_rf = random_forest_model(headers, teammate_x_other)
# header_model4_rf = random_forest_model(headers, opposition_x_other)

# Regular shots models
# regular_shots_model1 = logistic_model(regular_shots, basic_x)
#regular_shots_model2 = logistic_model(regular_shots, player_x_other)
#regular_shots_model3 = logistic_model(regular_shots, teammate_x_other)
regular_shots_model4 = logistic_model(regular_shots, opposition_x_other)
# regular_shots_model1_rf = random_forest_model(regular_shots, basic_x)
# regular_shots_model2_rf = random_forest_model(regular_shots, player_x_other)
# regular_shots_model3_rf = random_forest_model(regular_shots, teammate_x_other)
# regular_shots_model4_rf = random_forest_model(regular_shots, opposition_x_other)

regular_shots["our_xg"] = regular_shots_model4
# todo Recommendations
# todo clean up code
# - Test for multicollinearity - vif
# Compare my xg to statsbomb
xG_comparison = regular_shots[['player_name', 'goal', 'shot_statsbomb_xg', 'our_xg']]
xG_sum_per_player = xG_comparison.groupby('player_name').agg({
    'goal': 'sum',
    'shot_statsbomb_xg': 'sum',
    'our_xg': 'sum'
}).reset_index()


# Calculate absolute errors
xG_comparison['abs_error_statsbomb'] = np.abs(xG_comparison['goal'] - xG_comparison['shot_statsbomb_xg'])
xG_comparison['abs_error_our_xg'] = np.abs(xG_comparison['goal'] - xG_comparison['our_xg'])

# Calculate Mean Absolute Error (MAE)
mae_statsbomb = xG_comparison['abs_error_statsbomb'].mean()
mae_our_xg = xG_comparison['abs_error_our_xg'].mean()

# Calculate Root Mean Squared Error (RMSE)
rmse_statsbomb = np.sqrt((xG_comparison['abs_error_statsbomb']**2).mean())
rmse_our_xg = np.sqrt((xG_comparison['abs_error_our_xg']**2).mean())

print(f"Mean Absolute Error (MAE):")
print(f"StatsBomb xG: {mae_statsbomb}")
print(f"Our xG: {mae_our_xg}")

print(f"\nRoot Mean Squared Error (RMSE):")
print(f"StatsBomb xG: {rmse_statsbomb}")
print(f"Our xG: {rmse_our_xg}")

# https://www.youtube.com/watch?v=zM4VZR0px8E
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
