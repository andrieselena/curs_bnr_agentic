# Instrucțiuni Chatbot - Aplicație Curs BNR EUR/RON

## Scop

- Adăugare chatbot bazat pe un model LLM în aplicația Streamlit.
- Chatbotul trebuie să răspundă în limba română la întrebări despre cursul EUR/RON.
- Chatbotul trebuie să folosească tool-uri locale pentru a obține date din backend.
- Chatbotul trebuie să fie integrat vizibil în frontend, într-un widget de chat.
- Chatbotul trebuie să folosească backend-ul FastAPI existent pentru date și prognoze.

## Fișiere cheie

- `app_streamlit.py`: pagină Chatbot în UI și import `run_bnr_agent`.
- `src/curs_bnr/model_utils.py`: implementare tool-uri locale, registru `TOOL_REGISTRY`, funcția `choose_tool_for_message` și agentul `run_bnr_agent`.
- `src/curs_bnr/api/endpoints.py`: endpoint-ul `GET /api/forecast/latest` și alte API-uri utilizate de aplicație.

## Cum funcționează

1. Utilizatorul scrie o întrebare în pagina Chatbot din Streamlit.
2. `run_bnr_agent()` din `src/curs_bnr/model_utils.py` parcurge mesajul și alege un tool potrivit.
3. Tool-ul este executat local și obține date din backend.
4. Rezultatul tool-ului este trimis către modelul Gemini pentru a genera un răspuns natural în română.

## Răspunsuri recomandate

- "Care este ultima prognoză EUR/RON?"
- "Ce model este câștigător și ce eroare are?"
- "Care este ultimul curs EUR/RON disponibil?"
- "Cum pornesc dashboard-urile Optuna?"

## Backend local

Backend-ul FastAPI rulează local la:

```text
http://127.0.0.1:7772
```

De acolo, tool-urile pot apela endpoint-uri precum:

- `/api/rates`
- `/api/metrics/latest`
- `/api/forecast/latest`
- `/api/runs`
- `/api/scrape`
- `/api/optuna`

## Ce se verifică

- pagina Chatbot este prezentă în `app_streamlit.py`
- importul `from src.curs_bnr.model_utils import run_bnr_agent` este prezent
- `get_forecast_summary()` este definit și expus în `TOOL_REGISTRY`
- `choose_tool_for_message()` recunoaște `prognoză`, `predicție`, `forecast`
- `run_bnr_agent()` folosește tool-ul și generează răspunsul final
