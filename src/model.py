"""
The following file ...
"""
# Importing packages
import pickle
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, roc_curve, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Opening pickle files for each shot type
with open('../data_logos/headers.pkl', 'rb') as file:
    headers = pickle.load(file)

with open('../data_logos/regular_shots.pkl', 'rb') as file:
    regular_shots = pickle.load(file)

# Examining shape for test/train splits
print("Headers", headers.shape)
print("Regular", regular_shots.shape)

print(sum(regular_shots['goal_smf']) / len(regular_shots['goal_smf']))

# Setting independent variables (x), dependent variable (y), and encode lists for models
basic_x = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League']
player_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                  'technique_name', 'shot_first_time']
teammate_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                    'technique_name', 'shot_first_time',
                    'play_pattern_name', 'assist_type']
opposition_x_other = ['distance', 'angle', 'inverse_distance', 'inverse_angle', 'League',
                      'technique_name', 'shot_first_time',
                      'play_pattern_name', 'assist_type',
                      'under_pressure', 'shot_deflected']

basic_encode = ['League']
player_encode = ['League', 'technique_name', 'shot_first_time']
team_encode = ['League', 'technique_name', 'shot_first_time', 'play_pattern_name', 'assist_type']
opp_encode = ['League', 'technique_name', 'shot_first_time', 'play_pattern_name',
              'assist_type', 'under_pressure', 'shot_deflected']


# todo change to "league"
def splits_by_league(data, test_size=0.1):
    """
    Splits the data by league into training, validation, and test sets.

    Parameters:
       - data (pd.DataFrame): Dataframe containing input data for model and data splits
       - test_size (int): Percentage of overall dataframe to be split into the test data

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

        y_league = league_data['goal_smf']

        # Split the data
        train_data, test_data = train_test_split(league_data, test_size=test_size, random_state=123, stratify=y_league)

        # Append to respective lists
        test_frames.append(test_data)
        train_frames.append(train_data)

    # Concatenate all test and train frames
    train_data = pd.concat(train_frames)
    test_data = pd.concat(test_frames)

    return train_data, test_data


def logistic_model(data, x, encode):
    """
    Fits a logistic regression model with L2 regularization and evaluates its performance.

    Parameters:
    - data (pd.DataFrame): Dataframe containing input data for model and data splits
    - x: The features used as IVs to predict Y
    - encode: Variables that require one hot encoding

    Returns:
    - xg_values_logistic (np.array): Predicted probabilities for the data.
    """
    # Loading in and sorting the data
    train_data, test_data = splits_by_league(data, test_size=0.1)
    x_vars = x
    y_var = 'goal_smf'

    x_train = train_data[x_vars].copy()
    y_train = train_data[y_var]
    x_test = test_data[x_vars].copy()
    y_test = test_data[y_var]

    # One hot encoding for categorical variables
    x_train = pd.get_dummies(x_train, columns=encode, drop_first=True)
    x_test = pd.get_dummies(x_test, columns=encode, drop_first=True)

    # Dataset manipulation
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    # Regularising and running the model
    penalty = 'l2'
    C = 0.001

    model = LogisticRegression(penalty=penalty, C=C, max_iter=10000)
    model.fit(x_train_scaled,
              y_train)

    # Cross validation scores
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = cross_val_score(model, x_test_scaled, y_test, cv=kf, scoring='roc_auc')
    mean_cv_auc = cv_auc_scores.mean()
    print(f"Train Cross-Validation AUC Scores: {cv_auc_scores}")
    print(f"Mean Train Cross-Validation AUC Score: {mean_cv_auc}")

    # Brier scores
    brier_scorer = make_scorer(brier_score_loss, needs_proba=True)
    cv_brier_scores = cross_val_score(model, x_test_scaled, y_test, cv=kf, scoring=brier_scorer)
    mean_cv_brier = cv_brier_scores.mean()
    print(f"Cross-Validation Brier Scores: {cv_brier_scores}")
    print(f"Mean Cross-Validation Brier Score: {mean_cv_brier}")

    # RMSE scores
    rmse_scorer = make_scorer(mean_squared_error, squared=False, needs_proba=True)
    cv_rmse_scores = cross_val_score(model, x_test_scaled, y_test, cv=kf, scoring=rmse_scorer)
    mean_cv_rmse = cv_rmse_scores.mean()
    print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"Mean Cross-Validation RMSE Score: {mean_cv_rmse}")

    # Test set evaluation
    y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    x_data_scaled = pd.concat([x_train_scaled, x_test_scaled])
    xg_values_logistic = model.predict_proba(x_data_scaled)[:, 1]

    return xg_values_logistic


def random_forest_model(data, x, encode):
    """
    Fits a random forest model and evaluates its performance

     Parameters:
    - data (pd.Dataframe): Dataframe containing input data for model and data splits
    - x: The features used as IVs to predict Y
    - encode: Variables that require one hot encoding

    Returns:
    - Predicted probabilities for the test data.
    """
    x_vars = x
    y_var = 'goal_smf'
    train_data, test_data = splits_by_league(data, test_size=0.1)

    x_train = train_data[x_vars]
    y_train = train_data[y_var]
    x_test = test_data[x_vars]
    y_test = test_data[y_var]

    print(sum(y_train) / len(y_train))
    print(sum(y_test) / len(y_test))

    # One hot encoding for categorical variables
    x_train = pd.get_dummies(x_train, columns=encode, drop_first=True)
    x_test = pd.get_dummies(x_test, columns=encode, drop_first=True)

    # Dataset manipulation
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    # Running model
    classifier = RandomForestClassifier(n_estimators=100,
                                        warm_start=True,
                                        random_state=42,
                                        max_depth=12,
                                        min_samples_split=6
    )

    # Arrays to store the results
    train_accuracy = []
    test_accuracy = []
    train_error = []
    test_error = []

    # Loop for learning curve
    for i in range(1, 201):
        classifier.n_estimators = i
        classifier.fit(x_train_scaled, y_train)

        y_train_pred = classifier.predict(x_train_scaled)
        y_test_pred = classifier.predict(x_test_scaled)

        train_accuracy.append(accuracy_score(y_train, y_train_pred))
        test_accuracy.append(accuracy_score(y_test, y_test_pred))

        train_error.append(1 - accuracy_score(y_train, y_train_pred))
        test_error.append(1 - accuracy_score(y_test, y_test_pred))

    # Plotting the learning curves
    fig, axs = plt.subplots(2, figsize=(10, 12))

    # Plotting training history - RMSE
    axs[0].plot(train_accuracy, label='train rmse')
    axs[0].plot(train_accuracy, label='test rmse')
    axs[0].set_title("RMSE with number of trees")
    axs[0].set_xlabel("Number of Trees")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()

    # Plotting training history - Error
    axs[1].plot(train_error, label='train error')
    axs[1].plot(test_error, label='test error')
    axs[1].set_title("Error with number of trees")
    axs[1].set_xlabel("Number of Trees")
    axs[1].set_ylabel("Error")
    axs[1].legend()

    plt.show()

    # Cross-validation scores
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring='roc_auc')
    mean_cv_auc = cv_auc_scores.mean()
    print(f"Cross-Validation AUC Scores: {cv_auc_scores}")
    print(f"Mean Cross-Validation AUC Score: {mean_cv_auc}")

    # Brier scores
    brier_scorer = make_scorer(brier_score_loss, needs_proba=True)
    cv_brier_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring=brier_scorer)
    mean_cv_brier = cv_brier_scores.mean()
    print(f"Cross-Validation Brier Scores: {cv_brier_scores}")
    print(f"Mean Cross-Validation Brier Score: {mean_cv_brier}")

    # RMSE scores
    rmse_scorer = make_scorer(mean_squared_error, squared=False, needs_proba=True)
    cv_rmse_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring=rmse_scorer)
    mean_cv_rmse = cv_rmse_scores.mean()
    print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"Mean Cross-Validation RMSE Score: {mean_cv_rmse}")

    # Test set evaluation
    y_pred_prob = classifier.predict_proba(x_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Combining training and test data for final xG prediction
    x_data_scaled = pd.concat([x_train_scaled, x_test_scaled])
    xg_values_rf = classifier.predict_proba(x_data_scaled)[:, 1]

    return xg_values_rf


def xGBoost(data, x, encode):  # todo add type hinting
    x_vars = x
    y_var = 'goal_smf'
    train_data, test_data = splits_by_league(data, test_size=0.1)

    x_train = train_data[x_vars]
    y_train = train_data[y_var]
    x_test = test_data[x_vars]
    y_test = test_data[y_var]

    print(sum(y_train) / len(y_train))
    print(sum(y_test) / len(y_test))

    # One hot encoding for categorical variables
    x_train = pd.get_dummies(x_train, columns=encode, drop_first=True)
    x_test = pd.get_dummies(x_test, columns=encode, drop_first=True)

    # Dataset manipulation
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    # Running model
    classifier = XGBClassifier(objective='binary:logistic',
                               seed=42,
                               gamma=0,
                               learn_rate=0.1,
                               max_depth=4,
                               reg_lambda=1.0,
                               scale_pos_weight=5,
                               subsample=0.5,
                               colsample_bytree=0.5,
                               n_estimators=200)

    eval_set = [(x_train_scaled, y_train), (x_test_scaled, y_test)]

    fit_params = {
        "eval_set": eval_set,
        "early_stopping_rounds": 10,
        "verbose": True,
        "eval_metric": ["error", "rmse"]
    }
    # Fitting the model with early stopping
    classifier.fit(x_train_scaled,
                   y_train,
                   **fit_params)

    results = classifier.evals_result()

    fig, axs = plt.subplots(2, figsize=(10, 12))

    # Plotting training history - error
    axs[0].plot(results['validation_0']['rmse'], label='train rmse')
    axs[0].plot(results['validation_1']['rmse'], label='test rmse')
    axs[0].set_title("RMSE at each epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()

    # Plotting training history - log loss
    axs[1].plot(results['validation_0']['error'], label='train error')
    axs[1].plot(results['validation_1']['error'], label='test error')
    axs[1].set_title("Error at each epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend()

    plt.show()

    # Cross-validation scores
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring='roc_auc')
    mean_cv_auc = cv_auc_scores.mean()
    print(f"Cross-Validation AUC Scores: {cv_auc_scores}")
    print(f"Mean Cross-Validation AUC Score: {mean_cv_auc}")

    # Brier scores
    brier_scorer = make_scorer(brier_score_loss, needs_proba=True)
    cv_brier_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring=brier_scorer)
    mean_cv_brier = cv_brier_scores.mean()
    print(f"Cross-Validation Brier Scores: {cv_brier_scores}")
    print(f"Mean Cross-Validation Brier Score: {mean_cv_brier}")

    # RMSE scores
    rmse_scorer = make_scorer(mean_squared_error, squared=False, needs_proba=True)
    cv_rmse_scores = cross_val_score(classifier, x_test_scaled, y_test, cv=kf, scoring=rmse_scorer)
    mean_cv_rmse = cv_rmse_scores.mean()
    print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"Mean Cross-Validation RMSE Score: {mean_cv_rmse}")

    # Test set evaluation
    y_pred_prob = classifier.predict_proba(x_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Combining training and test data for final xG prediction
    x_data_scaled = pd.concat([x_train_scaled, x_test_scaled])
    xg_values_xgbst = classifier.predict_proba(x_data_scaled)[:, 1]

    return xg_values_xgbst


# Header models
# header_model1 = logistic_model(headers, basic_x, basic_encode)
# header_model2 = logistic_model(headers, player_x_other, player_encode)
# header_model3 = logistic_model(headers, teammate_x_other, team_encode)
# header_model4 = logistic_model(headers, opposition_x_other, opp_encode)
# header_model1_rf = random_forest_model(headers, basic_x, basic_encode)
# header_model2_rf = random_forest_model(headers, player_x_other, player_encode)
header_model3_rf = random_forest_model(headers, teammate_x_other, team_encode)
# header_model4_rf = random_forest_model(headers, opposition_x_other, opp_encode)
# header_model1_xgbst = xGBoost(regular_shots, basic_x, basic_encode)
# header_model2_xgbst = xGBoost(regular_shots, player_x_other, player_encode)
# header_model3_xgbst = xGBoost(regular_shots, teammate_x_other, team_encode)
# header_model4_xgbst = xGBoost(all_shots, opposition_x_other, opp_encode)


# Regular shots models
# regular_shots_model1 = logistic_model(regular_shots, basic_x, basic_encode)
# regular_shots_model2 = logistic_model(regular_shots, player_x_other, player_encode)
# regular_shots_model3 = logistic_model(regular_shots, teammate_x_other, team_encode)
# regular_shots_model4 = logistic_model(regular_shots, opposition_x_other, opp_encode)
# regular_shots_model1_rf = random_forest_model(regular_shots, basic_x, basic_encode)
# regular_shots_model2_rf = random_forest_model(regular_shots, player_x_other, player_encode)
# regular_shots_model3_rf = random_forest_model(regular_shots, teammate_x_other, team_encode)
# regular_shots_model4_rf = random_forest_model(regular_shots, opposition_x_other, opp_encode)
# regular_shots_model1_xgbst = xGBoost(regular_shots, basic_x, basic_encode)
# regular_shots_model2_xgbst = xGBoost(regular_shots, player_x_other, player_encode)
# regular_shots_model3_xgbst = xGBoost(regular_shots, teammate_x_other, team_encode)
regular_shots_model4_xgbst = xGBoost(regular_shots, opposition_x_other, opp_encode)

# Applying model xG output and saving to pickle
regular_shots["our_xg"] = regular_shots_model4_xgbst(regular_shots)
regular_shots.reset_index(drop=True, inplace=True)
with open('../data_logos/regular_shots_output.pkl', 'wb') as file:
    pickle.dump(regular_shots, file)

headers["our_xg"] = header_model3_rf(headers)
headers.reset_index(drop=True, inplace=True)
with open('../data_logos/headed_shots_output.pkl', 'wb') as file:
    pickle.dump(headers, file)

def statsbomb_comparison(shot_type):

#todo add type hint
    # Compare my xg to statsbomb
    xG_comparison = shot_type[['player_name', 'goal', 'shot_statsbomb_xg', 'our_xg']]

    # Calculate absolute errors
    xG_comparison['abs_error_statsbomb'] = np.abs(
    xG_comparison['goal'] - xG_comparison['shot_statsbomb_xg'])
    xG_comparison['abs_error_our_xg'] = np.abs(xG_comparison['goal'] - xG_comparison['our_xg'])

    # Calculate Mean Absolute Error (MAE)
    mae_statsbomb = xG_comparison['abs_error_statsbomb'].mean()
    mae_our_xg = xG_comparison['abs_error_our_xg'].mean()

    # Calculate Root Mean Squared Error (RMSE)
    rmse_statsbomb = np.sqrt((xG_comparison['abs_error_statsbomb'] ** 2).mean())
    rmse_our_xg = np.sqrt((xG_comparison['abs_error_our_xg'] ** 2).mean())

    print(f"Mean Absolute Error (MAE):")
    print(f"StatsBomb xG: {mae_statsbomb}")
    print(f"Our xG: {mae_our_xg}")

    print(f"\nRoot Mean Squared Error (RMSE):")
    print(f"StatsBomb xG: {rmse_statsbomb}")
    print(f"Our xG: {rmse_our_xg}")
