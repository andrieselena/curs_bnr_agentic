"""
run_optuna_optimization.py

Script dedicat pentru optimizarea hiperparametrilor folosind Optuna.
Rulează secvențial studiile pentru XGBoost, SARIMA și Prophet, 
salvându-le în baze de date SQLite separate.
"""

import sys
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.curs_bnr.config import CSV_PATH, VALUE_COL, OPTUNA_STUDIES_DIR
from src.curs_bnr.features.engineering import prepare_ml_data
from src.curs_bnr.training.trainers import train_sarima, train_xgboost


def run_xgboost_study(train_xgb: pd.DataFrame, db_path: str, n_trials: int = 15) -> optuna.Study:
    """Rulează studiul Optuna pentru modelul XGBoost."""
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
        tscv = TimeSeriesSplit(n_splits=3)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(train_xgb):
            train_fold = train_xgb.iloc[train_idx]
            val_fold = train_xgb.iloc[val_idx]
            
            model = train_xgboost(train_fold, value_col=VALUE_COL, xgb_params=params)
            
            X_val = val_fold.drop(columns=["Data", VALUE_COL])
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(val_fold[VALUE_COL].values, preds))
            fold_scores.append(rmse)
            
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        study_name="xgboost_optimization",
        storage=db_path,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def run_sarima_study(train_base: pd.DataFrame, db_path: str, n_trials: int = 10) -> optuna.Study:
    """Rulează studiul Optuna pentru modelul SARIMA."""
    def objective(trial: optuna.Trial) -> float:
        p = trial.suggest_int("p", 0, 2)
        d = trial.suggest_int("d", 0, 1)
        q = trial.suggest_int("q", 0, 2)
        
        P = trial.suggest_int("P", 0, 1)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 1)
        s = 5
        
        tscv = TimeSeriesSplit(n_splits=3)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(train_base):
            train_fold = train_base.iloc[train_idx]
            val_fold = train_base.iloc[val_idx]
            
            try:
                fitted_model = train_sarima(
                    train_fold, 
                    value_col=VALUE_COL, 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                predictions = fitted_model.forecast(steps=len(val_fold))
                rmse = np.sqrt(mean_squared_error(val_fold[VALUE_COL].values, predictions.values))
                fold_scores.append(rmse)
            except Exception:
                raise optuna.exceptions.TrialPruned()
                
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        study_name="sarima_optimization",
        storage=db_path,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def run_prophet_study(train_base: pd.DataFrame, db_path: str, n_trials: int = 10) -> optuna.Study:
    """Rulează studiul Optuna pentru modelul Prophet."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True)
        }
        
        df_prophet = train_base[["Data", VALUE_COL]].rename(columns={"Data": "ds", VALUE_COL: "y"})
        tscv = TimeSeriesSplit(n_splits=3)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(df_prophet):
            train_fold = df_prophet.iloc[train_idx]
            val_fold = df_prophet.iloc[val_idx]
            
            try:
                model = Prophet(daily_seasonality=False, yearly_seasonality=True, **params)
                model.fit(train_fold)
                
                future = val_fold[["ds"]]
                forecast = model.predict(future)
                rmse = np.sqrt(mean_squared_error(val_fold["y"].values, forecast["yhat"].values))
                fold_scores.append(rmse)
            except Exception:
                raise optuna.exceptions.TrialPruned()
                
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        study_name="prophet_optimization",
        storage=db_path,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    """
    Orchestrează fluxul de optimizare Optuna pentru toate cele 3 modele.
    """
    print("Încărcare date pentru optimizare Optuna...")
    try:
        data_dict = prepare_ml_data(str(CSV_PATH), VALUE_COL)
    except Exception as e:
        print(f"Eroare la preprocesare: {e}", file=sys.stderr)
        return

    train_base = data_dict["train_base"]
    train_xgb = data_dict["train_xgb"]

    OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

    xgb_db_path = OPTUNA_STUDIES_DIR / "xgboost_optimization.db"
    sarima_db_path = OPTUNA_STUDIES_DIR / "sarima_optimization.db"
    prophet_db_path = OPTUNA_STUDIES_DIR / "prophet_optimization.db"

    xgb_db = f"sqlite:///{xgb_db_path.as_posix()}"
    sarima_db = f"sqlite:///{sarima_db_path.as_posix()}"
    prophet_db = f"sqlite:///{prophet_db_path.as_posix()}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results: Dict[str, Dict[str, Any]] = {}

    print("\n--- Optimizare XGBoost ---")
    study_xgb = run_xgboost_study(train_xgb, xgb_db, n_trials=10)
    results["XGBoost"] = {
        "best_params": study_xgb.best_params,
        "best_rmse": study_xgb.best_value,
        "db_path": str(xgb_db_path)
    }
    print(f"Finalizat XGBoost. Cel mai bun RMSE: {study_xgb.best_value:.4f}")

    print("\n--- Optimizare SARIMA ---")
    study_sarima = run_sarima_study(train_base, sarima_db, n_trials=5)
    results["SARIMA"] = {
        "best_params": study_sarima.best_params,
        "best_rmse": study_sarima.best_value,
        "db_path": str(sarima_db_path)
    }
    print(f"Finalizat SARIMA. Cel mai bun RMSE: {study_sarima.best_value:.4f}")

    print("\n--- Optimizare Prophet ---")
    study_prophet = run_prophet_study(train_base, prophet_db, n_trials=5)
    results["Prophet"] = {
        "best_params": study_prophet.best_params,
        "best_rmse": study_prophet.best_value,
        "db_path": str(prophet_db_path)
    }
    print(f"Finalizat Prophet. Cel mai bun RMSE: {study_prophet.best_value:.4f}")

    print("\n" + "="*50)
    print("REZULTATE FINALE OPTUNA")
    print("="*50)
    for model_name, res in results.items():
        print(f"Model: {model_name}")
        print(f"  Cel mai bun RMSE: {res['best_rmse']:.4f}")
        print(f"  Cei mai buni parametri: {res['best_params']}")
        print(f"  Bază de date SQLite: {res['db_path']}\n")


if __name__ == "__main__":
    main()
