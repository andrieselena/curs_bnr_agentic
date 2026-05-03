import sqlite3
from src.curs_bnr.database.connection import get_db_connection

def init_db() -> None:
    """Creează tabelele necesare în baza de date dacă acestea nu există deja."""
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS historical_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT UNIQUE NOT NULL,
        value REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        hyperparameters TEXT
    );

    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        mae REAL,
        rmse REAL,
        mape REAL,
        FOREIGN KEY (run_id) REFERENCES training_runs (id)
    );

    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        forecast_date TEXT NOT NULL,
        forecast_value REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES training_runs (id)
    );
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript(create_tables_sql)
        conn.commit()
