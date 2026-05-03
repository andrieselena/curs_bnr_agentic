import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from typing import (
    Any,
    Dict,
)

from src.curs_bnr.config import VALUE_COL


def evaluate_prophet(model: Any, test_base: pd.DataFrame) -> np.ndarray:
    """
    Generează predicțiile pentru setul de test folosind modelul Prophet.
    """
    future = test_base[["Data"]].rename(columns={"Data": "ds"})
    forecast = model.predict(future)
    return forecast["yhat"].values


def evaluate_xgboost(model: Any, test_xgb: pd.DataFrame, value_col: str) -> np.ndarray:
    """
    Generează predicțiile pentru setul de test folosind modelul XGBoost.
    """
    X_test = test_xgb.drop(columns=["Data", value_col])
    predictions = model.predict(X_test)
    return predictions


def evaluate_sarima(model: Any, test_base: pd.DataFrame) -> np.ndarray:
    """
    Generează predicțiile pentru setul de test folosind modelul SARIMA.
    """
    steps = len(test_base)
    predictions = model.forecast(steps=steps)
    return predictions.values


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculează metricile de eroare (MAE, RMSE, MAPE) între valorile reale și cele prezise.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def compare_and_save_best_model(
    models_dict: Dict[str, Any], 
    data_dict: Dict[str, pd.DataFrame], 
    value_col: str = VALUE_COL,
    save_path: str = "model_castigator.pkl"
) -> Dict[str, Any]:
    """
    Orchestrează procesul de evaluare comparativă a celor 3 modele.

    Calculează predicțiile și metricile de eroare pentru fiecare model folosind
    setul său specific de test. Compară modelele bazat pe eroarea RMSE și 
    îl salvează pe cel mai bun local folosind joblib.

    Args:
        models_dict (Dict[str, Any]): Dicționarul cu modelele antrenate.
        data_dict (Dict[str, pd.DataFrame]): Dicționarul cu subseturile de date.
        value_col (str): Denumirea coloanei ce conține cursul valutar.
        save_path (str): Calea fișierului pentru salvarea modelului câștigător.

    Returns:
        Dict[str, Any]: Structură clară conținând metricile, numele câștigătorului
                        și calea la care a fost salvat.
    """
    test_base = data_dict["test_base"]
    test_xgb = data_dict["test_xgb"]
    
    y_true_base = test_base[value_col].values
    y_true_xgb = test_xgb[value_col].values

    metrics = {}

    if "prophet" in models_dict:
        pred_prophet = evaluate_prophet(models_dict["prophet"], test_base)
        metrics["prophet"] = calculate_metrics(y_true_base, pred_prophet)

    if "xgboost" in models_dict:
        pred_xgb = evaluate_xgboost(models_dict["xgboost"], test_xgb, value_col)
        metrics["xgboost"] = calculate_metrics(y_true_xgb, pred_xgb)

    if "sarima" in models_dict:
        pred_sarima = evaluate_sarima(models_dict["sarima"], test_base)
        metrics["sarima"] = calculate_metrics(y_true_base, pred_sarima)

    best_model_name = ""
    best_rmse = float("inf")

    for model_name, model_metrics in metrics.items():
        if model_metrics["RMSE"] < best_rmse:
            best_rmse = model_metrics["RMSE"]
            best_model_name = model_name

    best_model = models_dict[best_model_name]
    joblib.dump(best_model, save_path)

    return {
        "metrics": metrics,
        "best_model_name": best_model_name,
        "save_path": save_path
    }
