import pandas as pd
from typing import Dict

from src.curs_bnr.config import VALUE_COL, TEST_SIZE_DAYS


def prepare_ml_data(csv_path: str, value_col: str = VALUE_COL) -> Dict[str, pd.DataFrame]:
    """
    Încarcă, validează, îmbogățește cu feature-uri și împarte setul de date.

    Ajustat pentru a returna seturi de date specifice nevoilor fiecărui model:
    - Setul de bază (doar seria cronologică curată) pentru Prophet și SARIMA.
    - Setul extins (cu variabile derivate/feature engineering) pentru XGBoost.

    Args:
        csv_path (str): Calea către fișierul CSV de intrare.
        value_col (str): Denumirea coloanei ce conține cursul valutar.

    Returns:
        Dict[str, pd.DataFrame]: Un dicționar ce conține cheile:
            - 'train_base': Set de antrenament pentru Prophet/SARIMA
            - 'test_base': Set de test pentru Prophet/SARIMA
            - 'train_xgb': Set de antrenament cu feature engineering pentru XGBoost
            - 'test_xgb': Set de test cu feature engineering pentru XGBoost

    Raises:
        ValueError: Dacă datele sunt invalide (ex: valori nule, non-numerice, lipsă coloane).
    """
    df = pd.read_csv(csv_path)

    required_columns = ["Data", value_col]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coloana obligatorie '{col}' nu a fost găsită în fișierul CSV.")

    if df["Data"].isnull().any():
        raise ValueError("Coloana 'Data' conține valori nule.")
        
    if df[value_col].isnull().any():
        raise ValueError(f"Coloana '{value_col}' conține valori nule.")

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Coloana '{value_col}' trebuie să fie de tip numeric.")

    df["Data"] = pd.to_datetime(df["Data"])
    df_base = df.sort_values(by="Data", ascending=True).reset_index(drop=True)

    df_xgb = df_base.copy()
    
    df_xgb["DayOfWeek"] = df_xgb["Data"].dt.dayofweek
    df_xgb["Month"] = df_xgb["Data"].dt.month
    
    df_xgb["Lag_1"] = df_xgb[value_col].shift(1)
    df_xgb["Lag_2"] = df_xgb[value_col].shift(2)
    df_xgb["Lag_3"] = df_xgb[value_col].shift(3)
    
    df_xgb["MA_7"] = df_xgb[value_col].rolling(window=7).mean()
    df_xgb["MA_14"] = df_xgb[value_col].rolling(window=14).mean()

    df_xgb = df_xgb.dropna().reset_index(drop=True)

    test_size = TEST_SIZE_DAYS
    
    if len(df_base) <= test_size or len(df_xgb) <= test_size:
        raise ValueError(f"Setul de date este prea mic pentru a extrage un test de {test_size} zile.")

    train_base = df_base.iloc[:-test_size].copy()
    test_base = df_base.iloc[-test_size:].copy()

    train_xgb = df_xgb.iloc[:-test_size].copy()
    test_xgb = df_xgb.iloc[-test_size:].copy()

    return {
        "train_base": train_base,
        "test_base": test_base,
        "train_xgb": train_xgb,
        "test_xgb": test_xgb
    }
