from typing import List, Tuple
import pandas as pd

from src.curs_bnr.config import CURRENCY


def format_and_validate_data(raw_data: List[Tuple[str, str]], currency: str = CURRENCY) -> pd.DataFrame:
    """
    Preia datele brute extrase de pe site, le curăță, le formatează și le validează.

    Convertește coloana de dată în obiecte datetime, corectează separatorul
    zecimal pentru valori și le convertește în float. Sortează datele cronologic
    crescător și aplică validări de integritate.

    Args:
        raw_data (List[Tuple[str, str]]): Datele extrase de funcția de scraping.
        currency (str): Valuta procesată, folosită pentru a denumi coloana (ex: 'EUR').

    Returns:
        pd.DataFrame: Un DataFrame curățat, sortat și validat.

    Raises:
        ValueError: Dacă datele eșuează oricare dintre pașii de validare 
                    (valori nule, valori ne-pozitive, duplicate etc.).
    """
    if not raw_data:
        raise ValueError("Datele brute de intrare sunt goale.")

    value_col_name = f"Valoare_{currency}"
    df = pd.DataFrame(raw_data, columns=["Data", value_col_name])

    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")

    df[value_col_name] = df[value_col_name].astype(str).str.replace(",", ".", regex=False)
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")

    df = df.sort_values(by="Data", ascending=True).reset_index(drop=True)
    
    if df.isnull().values.any():
        raise ValueError("Setul de date conține valori nule sau conversia a eșuat.")

    if not (df[value_col_name] > 0).all():
        raise ValueError("Setul de date conține valori ale cursului mai mici sau egale cu zero.")

    if df.duplicated(subset=["Data"]).any():
        raise ValueError("Setul de date conține duplicate calendaristice.")

    return df
