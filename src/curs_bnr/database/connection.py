import sqlite3
from contextlib import contextmanager
from typing import Generator

from src.curs_bnr.config import APP_DB_PATH


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Returnează o conexiune la baza de date SQLite, folosind context manager."""
    conn = sqlite3.connect(str(APP_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
