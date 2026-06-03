# Plan Implementare Chatbot

## Obiectiv

Implementarea unui chatbot integrat în aplicația Streamlit care folosește un agent LLM și tool-uri locale pentru a răspunde la întrebări despre cursul EUR/RON.

## Componente principale

1. `app_streamlit.py`
   - Adăugarea unei pagini "Chatbot" în meniul lateral.
   - Crearea unui widget `st.chat_input` pentru introducerea întrebărilor.
   - Afișarea chat-ului în interfață cu istoric de conversație.
   - Import `from src.curs_bnr.model_utils import run_bnr_agent`.

2. `src/curs_bnr/model_utils.py`
   - Definirea funcțiilor de tool local:
     - `get_latest_exchange_rate()`
     - `get_forecast_summary()`
     - `get_model_metrics()`
     - `get_training_runs()`
     - `trigger_scrape()`
     - `get_optuna_info()`
   - Definirea `TOOL_REGISTRY` cu tool-ul `get_forecast_summary`.
   - Definirea `choose_tool_for_message()` care selectează `get_forecast_summary` când mesajul conține "prognoză", "predicție", "forecast".
   - Implementarea `run_bnr_agent()` pentru a rula un tool local, a transmite rezultatul către Gemini și a returna un răspuns natural.

3. `src/curs_bnr/api/endpoints.py`
   - Verificarea endpoint-ului `GET /api/forecast/latest` pentru ultima prognoză.
   - Asigurarea că backend-ul returnează JSON valid pentru tool-urile chatbot.

## Flux

- Utilizatorul scrie întrebarea în Streamlit.
- `run_bnr_agent()` identifică tool-ul potrivit.
- Tool-ul local este apelat și obține date din backend.
- Agentul LLM generează răspunsul final pe baza rezultatului tool-ului.

## Validări

- Pagina Chatbot funcționează în `app_streamlit.py`.
- Tool-ul `get_forecast_summary` este apelabil și returnează date valide.
- `choose_tool_for_message()` răspunde corect la expresii legate de prognoză.
- Endpoint-ul `/api/forecast/latest` răspunde cu date sau mesaj clar.
