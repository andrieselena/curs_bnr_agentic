from typing import Any, Dict, Optional, Tuple

import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.curs_bnr.config import VALUE_COL


def train_prophet(train_base: pd.DataFrame, value_col: str = VALUE_COL) -> Any:
    """
    Antrenează modelul Prophet pe setul de date de bază.

    Prophet necesită formatarea specifică a dataframe-ului, redenumind 
    coloanele în 'ds' (pentru dată) și 'y' (pentru valoarea numerică).
    Este folosit pentru a capta automat trendurile și sezonalitățile.

    Args:
        train_base (pd.DataFrame): Setul de antrenament de bază (fără feature engineering).
        value_col (str): Denumirea coloanei ce conține cursul valutar.

    Returns:
        Any: Obiectul modelului Prophet antrenat.
    """
    df_prophet = train_base[["Data", value_col]].rename(columns={
        "Data": "ds", 
        value_col: "y"
    })
    
    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(df_prophet)
    
    return model


def train_xgboost(
    train_xgb: pd.DataFrame, 
    value_col: str = VALUE_COL, 
    xgb_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Antrenează modelul XGBoost Regressor pe setul de date extins.

    Folosește toate variabilele derivate (lags, medii mobile). Suportă
    injectarea parametrilor (xgb_params) pentru a facilita optimizarea
    ulterioară (ex: prin Optuna).

    Args:
        train_xgb (pd.DataFrame): Setul de antrenament cu variabile derivate adăugate.
        value_col (str): Denumirea coloanei țintă ce trebuie previzionată.
        xgb_params (dict, optional): Parametri de tuning. Dacă este None, se folosesc
                                     parametri standard de bază.

    Returns:
        Any: Obiectul modelului XGBoostRegressor antrenat.
    """
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": 42,
            "objective": "reg:squarederror"
        }
        
    X_train = train_xgb.drop(columns=["Data", value_col])
    y_train = train_xgb[value_col]
    
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    return model


def train_sarima(
    train_base: pd.DataFrame, 
    value_col: str = VALUE_COL, 
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 5)
) -> Any:
    """
    Antrenează modelul statistic SARIMA pe setul de date de bază.

    Ajustează modelul folosind argumentele oferite. Lipsa parametrizării rigide 
    facilitează posibilitatea de căutare pe grid a parametrilor (p,d,q).

    Args:
        train_base (pd.DataFrame): Setul de antrenament de bază.
        value_col (str): Denumirea coloanei ce conține cursul valutar.
        order (Tuple[int,int,int]): Parametrii autoregresivi și de medie mobilă (p, d, q).
        seasonal_order (Tuple[int,int,int,int]): Parametrii sezonali (P, D, Q, s).

    Returns:
        Any: Obiectul modelului ajustat (Fitted Model) din statsmodels.
    """
    endog = train_base[value_col]
    
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    
    return fitted_model


def train_all_models(data_dict: Dict[str, pd.DataFrame], value_col: str = VALUE_COL) -> Dict[str, Any]:
    """
    Orchestrează antrenarea celor 3 modele folosind seturile de date corespunzătoare.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dicționarul produs de preprocesare_ml.py
                                             care conține cheile necesare.
        value_col (str): Denumirea coloanei ce conține cursul valutar.

    Returns:
        Dict[str, Any]: Un dicționar conținând instanțele celor 3 modele antrenate.
    """
    train_base = data_dict["train_base"]
    train_xgb = data_dict["train_xgb"]

    model_prophet = train_prophet(train_base, value_col)
    model_xgb = train_xgboost(train_xgb, value_col)
    model_sarima = train_sarima(train_base, value_col, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))

    return {
        "prophet": model_prophet,
        "xgboost": model_xgb,
        "sarima": model_sarima
    }
