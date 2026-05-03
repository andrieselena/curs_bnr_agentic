# Curs Valutar BNR - Prognoză EUR/RON

Acest proiect implementează un pipeline complet pentru extragerea, procesarea, antrenarea și optimizarea modelelor de prognoză pentru cursul valutar EUR/RON, folosind date de pe site-ul cursbnr.ro.

## Structură

- `src/curs_bnr/` - codul principal al proiectului
- `data/` - fișierul CSV cu datele istorice
- `outputs/models/` - modelul câștigător salvat
- `outputs/reports/` - grafice și rapoarte HTML
- `outputs/optuna_studies/` - baze de date SQLite pentru studiile Optuna
- `agentic_docs/` - documentație și prompturi folosite

## Instalare

```bash
pip install -r requirements.txt
```

## Rulare

Extragerea datelor:

```bash
python run_scraper.py
```

Antrenarea modelelor și generarea raportului vizual:

```bash
python run_training_pipeline.py
```

Optimizarea hiperparametrilor cu Optuna:

```bash
python run_optuna_optimization.py
```

## Vizualizare Optuna

Deschide notebook-ul:

```text
pornire_dashboard_optuna.ipynb
```

Rulează celula pentru modelul dorit și accesează în browser:

- XGBoost: `http://127.0.0.1:8090`
- SARIMA: `http://127.0.0.1:8091`
- Prophet: `http://127.0.0.1:8092`