from multiprocessing import Pool

import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope as ho_scope
import numpy as np
import pandas as pd

from utils import best_hyperparams_transform


def print_classif_report(model_name, pred, y):
    print(f'{model_name}: ')
    print("- roc_auc_score:", round(roc_auc_score(y.values.ravel(), pred), 4))
    print("- accuracy_score:", round(accuracy_score(y.values.ravel(), pred), 4))
    print("- f1_score:", round(f1_score(y.values.ravel(), pred), 4))
    print()


def model_fit_predict(space):
    eval_data = space['eval_data']
    model_type = space['model']
    params = space['params']

    model = model_type(**params)

    X_train = eval_data['X_train'] if type(eval_data['X_train']) == np.ndarray else eval_data['X_train'].values
    y_train = eval_data['y_train'] if type(eval_data['y_train']) == np.ndarray else eval_data['y_train'].values
    X_test = eval_data['X_test'] if type(eval_data['X_test']) == np.ndarray else eval_data['X_test'].values

    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)
    return model, pred


def objective(space):
    metric = space['metric']
    eval_data = space['eval_data']
    _, pred = model_fit_predict(space)

    curr_metric = metric(eval_data['y_test'].values.ravel(), pred)

    return -curr_metric


def hp_optimize(trial, space, max_evals):
    # trial, space, max_evals = param
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trial, max_evals=max_evals,
                rstate=np.random.default_rng(42))
    return best


def hyperparams_auto_tuning(params):
    with Pool() as p:
        result = p.starmap(hp_optimize, params)
        print(result)
    return result


def set_xgb_params(eval_data, metric):
    params_xgb = {
        'max_depth': ho_scope.int(hp.quniform("max_depth", 1, 20, 1)),
        # 'learning_rate': hp.uniform("learning_rate", 0.0005, 0.3),
        'min_child_weight': ho_scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
        'n_estimators': ho_scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    }

    space_xgb = {
        'params': params_xgb,
        'eval_data': eval_data,
        'metric': metric,
        'model': xgb.XGBClassifier
    }
    return space_xgb


def set_rf_params(eval_data, metric):
    params_rf = {
        'n_estimators': ho_scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        'max_depth': ho_scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'min_samples_leaf': ho_scope.int(hp.quniform('min_samples_leaf', 2, 10, 1)),
        'min_samples_split': ho_scope.int(hp.quniform('min_samples_split', 2, 10, 1))
    }

    space_rf = {
        'params': params_rf,
        'eval_data': eval_data,
        'metric': metric,
        'model': RandomForestClassifier
    }
    return space_rf


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    model = joblib.load(path)
    return model


def tuning(tuning, eval_data, metric, max_evals, save_name, load_name):
    if tuning:
        # Гиперпараметры для XGBoost и RandomForest
        space_rf = set_rf_params(eval_data, metric)
        space_xgb = set_xgb_params(eval_data, metric)

        # Multiprocessing hyperopt training
        params_for_hp_tuning = ((Trials(), space_xgb, max_evals), (Trials(), space_rf, max_evals))

        # Подбор тюнинг гиперпараметров:
        result = hyperparams_auto_tuning(params_for_hp_tuning)

        # Обработка гиперпараметров:
        best_hyperparams_rf = best_hyperparams_transform(result[1])
        best_hyperparams_xgb = best_hyperparams_transform(result[0])

        # Обучение Random Forest и XGBoost с лучшими гиперпараметрами
        # random forest
        space = {
            'params':best_hyperparams_rf,
            'eval_data': eval_data,
            'model': RandomForestClassifier
        }

        rf_model, rf_pred = model_fit_predict(space)

        # xgboost
        space = {
            'params':best_hyperparams_xgb,
            'eval_data': eval_data,
            'model': xgb.XGBClassifier
        }

        xgb_model, xgb_pred = model_fit_predict(space)
        save_model(xgb_model, f'./models/xgboost{save_name}.pkl')
        save_model(rf_model, f'./models/random_forest{save_name}.pkl')

        return rf_model, rf_pred, xgb_model, xgb_pred

    else:
        xgb_model = load_model(f'./models/xgboost{load_name}.pkl')
        xgb_pred = xgb_model.predict(eval_data['X_test'])

        rf_model = load_model(f'./models/random_forest{load_name}.pkl')
        rf_pred = xgb_model.predict(eval_data['X_test'])

        return rf_model, rf_pred, xgb_model, xgb_pred
