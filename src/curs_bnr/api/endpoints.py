from typing import List, Dict, Any
from fastapi import APIRouter

from src.curs_bnr.api.schemas import RateResponse, MetricResponse, RunResponse, ForecastResponse, MessageResponse
from src.curs_bnr.database.connection import get_db_connection
from src.curs_bnr.database.crud import populate_historical_rates_from_csv
from src.curs_bnr.data.scraper import extract_raw_data
from src.curs_bnr.data.processing import format_and_validate_data
from src.curs_bnr.config import START_DATE, END_DATE, CURRENCY, CSV_PATH

router = APIRouter()

@router.get("/api/rates", response_model=List[RateResponse])
def get_historical_rates() -> List[Dict[str, Any]]:
    """Returnează istoricul cursului EUR/RON din baza de date."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, date, value FROM historical_rates ORDER BY date ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

@router.get("/api/metrics/latest")
def get_latest_metrics() -> Any:
    """Returnează ultimele metrici disponibile, împreună cu modelul asociat."""
    query = """
    SELECT m.mae, m.rmse, m.mape, r.id as run_id, r.model_name, r.version, r.timestamp
    FROM metrics m
    JOIN training_runs r ON m.run_id = r.id
    ORDER BY r.timestamp DESC LIMIT 1
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        if row is None:
            return {"message": "Nu există metrici în baza de date. Rulați antrenarea mai întâi."}
        
        return dict(row)

@router.get("/api/runs", response_model=List[RunResponse])
def get_runs() -> List[Dict[str, Any]]:
    """Returnează lista rulărilor de antrenare."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, model_name, version, hyperparameters FROM training_runs ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

@router.get("/api/forecast/latest")
def get_latest_forecast() -> Any:
    """Returnează ultima prognoză disponibilă."""
    query = """
    SELECT f.id, f.run_id, f.forecast_date, f.forecast_value
    FROM forecasts f
    JOIN training_runs r ON f.run_id = r.id
    ORDER BY r.timestamp DESC, f.forecast_date ASC
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return {"message": "Nu există prognoze disponibile."}
        
        return [dict(r) for r in rows]

@router.post("/api/scrape", response_model=MessageResponse)
def scrape_data() -> Dict[str, str]:
    """Rulează scraping-ul și actualizează historical_rates în DB."""
    try:
        raw_data = extract_raw_data(CURRENCY, START_DATE, END_DATE)
        df_processed = format_and_validate_data(raw_data, CURRENCY)
        df_processed.to_csv(CSV_PATH, index=False)
        populate_historical_rates_from_csv()
        return {"message": f"Scraping complet și BD a fost actualizată: {START_DATE} - {END_DATE}."}
    except Exception as e:
        return {"message": f"Eroare la scraping: {e}"}

@router.post("/api/train", response_model=MessageResponse)
def train_model() -> Dict[str, str]:
    """Returnează un mesaj clar deoarece rularea completă blochează backend-ul."""
    return {
        "message": "Pentru antrenarea completă, rulați `python run_training_pipeline.py` manual din terminal."
    }

@router.post("/api/optuna", response_model=Dict[str, Any])
def run_optuna() -> Dict[str, Any]:
    """Returnează detaliile dashboard-urilor Optuna."""
    return {
        "message": "Optimizarea se rulează separat: `python run_optuna_optimization.py`.",
        "dashboards": {
            "XGBoost": "http://127.0.0.1:8080",
            "SARIMA": "http://127.0.0.1:8081",
            "Prophet": "http://127.0.0.1:8082"
        }
    }
