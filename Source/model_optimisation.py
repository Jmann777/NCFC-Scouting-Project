#todo add file description

# Importing packages
import pandas as pd
import model

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def xGBoost_optimisation(data, x, encode):
    #todo add type hints
    x_vars = x
    y_var = 'goal_smf'
    train_data, test_data = model.splits_by_league(data, test_size=0.1)

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
    classifier = XGBClassifier(objective='binary:logistic', seed=42)
    classifier.fit(x_train_scaled,
                   y_train)

    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        'gamma': [0, 0.25, 1.0],
        'reg_lambda': [1.0, 10.0, 100.0],
        'scale_pos_weight': [1, 3, 5]
    }

    optimal_params = GridSearchCV(
        estimator=XGBClassifier(objective='binary:logistic',
                               seed=42,
                               subsample=0.5,
                               colsample_bytree=0.5),
        param_grid=param_grid,
        scoring='roc_auc',
        verbose=0,
        n_jobs=10,
        cv=3
    )

    optimal_params.fit(x_train_scaled,
                       y_train)
    print(optimal_params.best_params_)

    return optimal_params

regular_shots_model4_xgbst = xGBoost_optimisation(model.regular_shots, model.opposition_x_other, model.opp_encode)

#todo create for rndm forest