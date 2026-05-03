# Plan de Refactorizare: Arhitectură Modulară "Curs BNR"

## 1. Structura Finală Propusă
```text
curs_bnr_agentic/
├── agentic_docs/                # Documentație (planuri, prompturi)
├── data/
│   └── raw/                     # Date primare descărcate (ex: curs_EUR_22-02...csv)
├── outputs/
│   ├── models/                  # Modele salvate (ex: model_castigator.pkl)
│   ├── reports/                 # Grafice HTML și rapoarte text
│   └── optuna_studies/          # Baze de date SQLite pentru Optuna Dashboard
├── src/
│   └── curs_bnr/                # Pachetul principal Python
│       ├── __init__.py
│       ├── data_collection/     # Module pentru extragere și procesare scraping
│       ├── features/            # Module pentru preprocesare ML și feature engineering
│       ├── models/              # Module pentru antrenare, evaluare și tuning
│       └── visualization/       # Module pentru generarea graficelor Plotly
├── config.py                    # Variabile globale și setări de mediu
├── main.py                      # Orchestratorul pentru Data Collection
├── main_ml.py                   # Orchestratorul pentru Machine Learning
├── pyproject.toml               # Configurația pachetului și a dependențelor
├── run_optuna_optimization.py   # Script dedicat exclusiv optimizării Optuna
└── pornire_dashboard_optuna.ipynb # Notebook interactiv pentru vizualizarea Optuna
```

## 2. Mutarea Fișierelor Python (Mapare)
Fișierele existente vor fi mutate logic în pachetul intern `src/curs_bnr/`:
*   `scraper.py` și `procesare.py` $\rightarrow$ `src/curs_bnr/data_collection/`
*   `preprocesare_ml.py` $\rightarrow$ `src/curs_bnr/features/`
*   `antrenare.py`, `evaluare.py` și `tuning.py` $\rightarrow$ `src/curs_bnr/models/`
*   `vizualizare.py` $\rightarrow$ `src/curs_bnr/visualization/`

## 3. Responsabilitatea Fiecărui Modul
*   **data_collection**: Conține exclusiv logica de scraping a BNR-ului, curățarea șirurilor de caractere și validarea internă a răspunsurilor HTTP.
*   **features**: Izolează manipularea datelor. Responsabil de crearea variabilelor derivate (feature engineering) și separarea temporală (Train/Test) specifică.
*   **models**: Conține algoritmica ML pură. Interfațează cu algoritmii (Prophet, XGBoost, SARIMA) decuplat complet de sursa datelor din exterior.
*   **visualization**: Modul strict vizual, dedicat construirii structurilor `plotly` pornind de la dicționare sau DataFrame-uri cu predicții.

## 4. Introducerea `config.py` Centralizat
Vom crea în rădăcina proiectului (lângă `main.py`) un fișier `config.py`.
*   Va izola constantele esențiale: `CURRENCY = "EUR"`, `START_DATE`, `END_DATE`.
*   Va utiliza librăria `pathlib` pentru a construi dinamic path-urile absolute către `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR` și `OPTUNA_DIR`.
*   Toate scripturile de intrare vor prelua coordonatele executării direct din acest fișier.

## 5. Menținerea Funcționalității Existente (EUR/RON)
*   **Fără Modificări de Logică**: Refactorizarea este exclusiv la nivel spațial. Niciun mecanism de funcționare ML nu se va modifica.
*   **Ajustare Importuri**: În `main.py` și `main_ml.py`, vom schimba liniile de forma `from scraper import...` cu `from src.curs_bnr.data_collection.scraper import...`.
*   Atributele vor fi citite automat din `config.py`, păstrând pipeline-ul testat funcțional 100%.

## 6. Organizarea Sistemului de Outputs
Sistemul de Output va deveni extrem de riguros:
*   `outputs/models/`: Directoriul unde funcțiile din `evaluare.py` vor salva fișierul `.pkl`.
*   `outputs/reports/`: Directoriul unde `vizualizare.py` va exporta graficul `grafic_prognoza_final.html`.
*   `outputs/optuna_studies/`: Spațiu unde se va crea o bază de date SQLite (`sqlite:///outputs/optuna_studies/optuna_eur.db`) utilă monitorizării studiilor pe parcursul antrenamentului.

## 7. Introducerea Scriptului `run_optuna_optimization.py`
Acesta devine un program executabil dedicat:
*   Extrage logica greoaie și consumatoare de timp a studiului bayesian din `main_ml.py`.
*   Inițializează conexiunea la baza de date SQLite, reia un studiu existent (suportă întrerupere/reluare) și execută exclusiv testarea parametrilor pentru XGBoost (sau orice alt model optimizabil asincron).

## 8. Introducerea Notebook-ului `pornire_dashboard_optuna.ipynb`
*   Va oferi un scurt manual interactiv (un fel de wrapper).
*   Va demonstra comanda CLI (`optuna-dashboard sqlite:///outputs/optuna_studies/optuna_eur.db`) sau va porni procesul local în celulele Notebook-ului, fiind foarte util în cadrul prezentărilor academice pentru inspecția interactivă a graficelor.

## 9. Actualizarea `agentic_docs`
*   Acest plan propus va fi stocat sub denumirea `agentic_docs/plan_refactorizare.md`.
*   Vom iniția un document `agentic_docs/prompts.txt` care să sumarizeze instrucțiunile strategice urmate de agent de-a lungul procesului, pentru justificarea implementării.
