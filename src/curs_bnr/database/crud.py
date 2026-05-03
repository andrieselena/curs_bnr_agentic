import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict

from src.curs_bnr.config import CSV_PATH, VALUE_COL
from src.curs_bnr.database.connection import get_db_connection


def populate_historical_rates_from_csv() -> None:
    """Încarcă datele din CSV-ul existent și le inserează în tabelul historical_rates."""
    if not CSV_PATH.exists():
        print(f"Fișierul {CSV_PATH} nu există. Nu se poate popula baza de date.", file=sys.stderr)
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Eroare la citirea CSV-ului: {e}", file=sys.stderr)
        return
    
    if "Data" not in df.columns or VALUE_COL not in df.columns:
        print(f"Coloanele necesare ('Data', '{VALUE_COL}') nu se află în CSV.", file=sys.stderr)
        return
        
    records = df[["Data", VALUE_COL]].values.tolist()
    
    insert_sql = """
    INSERT OR IGNORE INTO historical_rates (date, value)
    VALUES (?, ?)
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(insert_sql, records)
        conn.commit()
        print(f"Au fost procesate cu succes {len(records)} rânduri pentru tabelul historical_rates.")


def save_training_results_to_db(
    model_name: str, 
    version: str, 
    hyperparameters: str, 
    metrics: Dict[str, float], 
    forecast_dates: List[str], 
    forecast_values: List[float]
) -> None:
    """Salvează o rulare de antrenament, metricele și prognozele aferente în baza de date SQLite."""
    timestamp = datetime.now().isoformat()
    
    insert_run_sql = """
    INSERT INTO training_runs (timestamp, model_name, version, hyperparameters)
    VALUES (?, ?, ?, ?)
    """
    insert_metrics_sql = """
    INSERT INTO metrics (run_id, mae, rmse, mape)
    VALUES (?, ?, ?, ?)
    """
    insert_forecast_sql = """
    INSERT INTO forecasts (run_id, forecast_date, forecast_value)
    VALUES (?, ?, ?)
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Inserare în training_runs
        cursor.execute(insert_run_sql, (timestamp, model_name, version, hyperparameters))
        run_id = cursor.lastrowid
        
        # 2. Inserare în metrics
        mae = metrics.get("MAE", None)
        rmse = metrics.get("RMSE", None)
        mape = metrics.get("MAPE", None)
        cursor.execute(insert_metrics_sql, (run_id, mae, rmse, mape))
        
        # 3. Inserare în forecasts
        if len(forecast_dates) == len(forecast_values):
            forecast_records = [
                (run_id, str(f_date), float(f_val)) 
                for f_date, f_val in zip(forecast_dates, forecast_values)
            ]
            cursor.executemany(insert_forecast_sql, forecast_records)
            
        conn.commit()
