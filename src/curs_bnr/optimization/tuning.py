import itertools
from typing import (
    Any,
    Dict,
    Tuple,
)

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.curs_bnr.config import VALUE_COL


def optimize_sarima_gridsearch(
    train_base: pd.DataFrame, 
    value_col: str = VALUE_COL, 
    n_splits: int = 3
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float]:
    """
    VARIANTA CLASICĂ (Exhaustivă): GridSearch pentru SARIMA.

    Explorează sistematic atât structura auto-regresivă `order` (p, d, q) 
    cât și structura sezonieră `seasonal_order` (P, D, Q, s) validând cu TimeSeriesSplit.

    Args:
        train_base (pd.DataFrame): Datele de antrenament de bază.
        value_col (str): Denumirea coloanei ce conține cursul valutar.
        n_splits (int): Numărul de fold-uri pentru cross-validarea pe serii temporale.

    Returns:
        Tuple[Tuple, Tuple, float]: Returnează `best_order`, `best_seasonal` și `RMSE`.
    """
    endog = train_base[value_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    p_values = [0, 1]
    d_values = [0, 1]
    q_values = [0, 1]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    s_values = [5]
    
    best_score = float("inf")
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 1, 5)
    
    for p, d, q, P, D, Q, s in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        fold_scores = []
        
        try:
            for train_idx, val_idx in tscv.split(endog):
                train_fold, val_fold = endog[train_idx], endog[val_idx]
                
                model = SARIMAX(train_fold, order=order, seasonal_order=seasonal_order)
                fitted = model.fit(disp=False)
                
                predictions = fitted.forecast(steps=len(val_fold))
                rmse = np.sqrt(mean_squared_error(val_fold, predictions))
                fold_scores.append(rmse)
                
            avg_rmse = float(np.mean(fold_scores))
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_order = order
                best_seasonal = seasonal_order
        except Exception:
            continue
            
    return best_order, best_seasonal, best_score


def optimize_prophet_gridsearch(
    train_base: pd.DataFrame, 
    value_col: str = VALUE_COL, 
    n_splits: int = 3
) -> Tuple[Dict[str, float], float]:
    """
    VARIANTA CLASICĂ (Exhaustivă): GridSearch pentru Prophet.

    Testează sistematic hiperparametrii relevanți ai modelului Prophet.
    """
    df_prophet = train_base[["Data", value_col]].rename(columns={"Data": "ds", value_col: "y"})
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    param_grid = {  
        "changepoint_prior_scale": [0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.1, 1.0, 10.0]
    }
    
    keys, values = zip(*param_grid.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = float("inf")
    best_params = {}
    
    for params in grid_combinations:
        fold_scores = []
        try:
            for train_idx, val_idx in tscv.split(df_prophet):
                train_fold = df_prophet.iloc[train_idx]
                val_fold = df_prophet.iloc[val_idx]
                
                model = Prophet(daily_seasonality=False, yearly_seasonality=True, **params)
                model.fit(train_fold)
                
                future = val_fold[["ds"]]
                forecast = model.predict(future)
                
                rmse = np.sqrt(mean_squared_error(val_fold["y"].values, forecast["yhat"].values))
                fold_scores.append(rmse)
                
            avg_rmse = float(np.mean(fold_scores))
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = params
        except Exception:
            continue
            
    return best_params, best_score


def optimize_xgboost_optuna(
    train_xgb: pd.DataFrame, 
    value_col: str = VALUE_COL, 
    n_trials: int = 20, 
    n_splits: int = 3
) -> Dict[str, Any]:
    """
    VARIANTA INTELIGENTĂ (Bayesiană): Optuna pentru XGBoost.

    Este superioară GridSearch-ului pe modele complexe (precum XGBoost) datorită
    spațiului de căutare continuu și capacității de evaluare inteligentă TPE.
    """
    X = train_xgb.drop(columns=["Data", value_col]).values
    y = train_xgb[value_col].values
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "reg:squarederror",
            "random_state": 42
        }
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_scores.append(rmse)
            
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42
    
    return best_params
