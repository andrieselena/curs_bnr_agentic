"""
config.py

Setări și constante centralizate pentru proiectul de estimare a cursului BNR.
Acest fișier izolează "magic numbers" și rutele sistemului de fișiere pentru a fi 
utilizate de toate modulele din pachet.
"""

from pathlib import Path

# ==========================================
# CONSTANTE GENERALE ȘI DATE
# ==========================================
CURRENCY: str = "EUR"
VALUE_COL: str = "Valoare_EUR"
START_DATE: str = "22/02/2020"
END_DATE: str = "19/03/2026"

# ==========================================
# PARAMETRI PENTRU MACHINE LEARNING
# ==========================================
# Mărimea setului de testare (ultimele zile)
TEST_SIZE_DAYS: int = 14

# Parametri pentru Feature Engineering
LAGS: tuple[int, ...] = (1, 2, 3)
MOVING_AVERAGES: tuple[int, ...] = (7, 14)

# ==========================================
# CĂI (PATHS) ABSOLUTE ȘI RELATIVE
# ==========================================
# Presupunând că config.py este în src/curs_bnr/, BASE_DIR va fi rădăcina proiectului
BASE_DIR: Path = Path(__file__).resolve().parents[2]

# Calea fixă stabilită către fișierul de date brute
CSV_PATH: Path = BASE_DIR / "data" / "curs_EUR_22-02-2020_19-03-2026.csv"

# Directoare de Output
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
OPTUNA_STUDIES_DIR: Path = OUTPUTS_DIR / "optuna_studies"
APP_DB_PATH: Path = OUTPUTS_DIR / "app_database.db"
