# Plan Implementare Frontend / Backend

## Obiectiv

Documentarea arhitecturii full-stack a proiectului, care combină un frontend Streamlit cu un backend FastAPI și un pipeline de machine learning.

## Componente principale

1. Frontend Streamlit
   - `app_streamlit.py`
   - Pagini dedicate: Acasă / Prognoză, Antrenare, Optuna Dashboards, Chatbot.
   - Consuma endpoint-uri FastAPI pentru date istorice, metrici, prognoze și rulări.
   - Încarcă fișierul HTML `outputs/reports/grafic_prognoza_final.html` pentru vizualizarea prognozei.

2. Backend FastAPI
   - `run_backend.py`
   - `src/curs_bnr/api/endpoints.py`
   - Endpoint-uri pentru:
     - `GET /api/rates`
     - `GET /api/metrics/latest`
     - `GET /api/runs`
     - `GET /api/forecast/latest`
     - `POST /api/scrape`
     - `POST /api/train` (redirectează spre rularea terminalului)
     - `POST /api/optuna`

3. Pipeline de date și training
   - `run_scraper.py` și `src/curs_bnr/data/scraper.py`
   - `run_training_pipeline.py` pentru reantrenarea completă și generarea graficului final.
   - `src/curs_bnr/features/engineering.py` și `src/curs_bnr/visualization/plots.py`.

4. Chatbot integrat
   - `src/curs_bnr/model_utils.py` mută logica agentului și tool-urilor.
   - `app_streamlit.py` importă `run_bnr_agent()` și afișează pagina Chatbot.

## Flux de dezvoltare

- Start backend: `python run_backend.py`
- Start frontend: `streamlit run app_streamlit.py`
- Rulează scraping: `python run_scraper.py`
- Rulează training complet: `python run_training_pipeline.py`

## Observații

- Backend-ul oferă date din baza SQLite și nu rulează training blocking direct prin API.
- Frontend-ul oferă o interfață unificată cu vizualizări, dashboard-uri Optuna și chat.
- Documentația trebuie păstrată în `agentic_docs/` pentru cerința de fișiere instrucțiuni și planuri.
