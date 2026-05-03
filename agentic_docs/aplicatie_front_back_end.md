# Scop

- aplicație cu front end și back end pentru prognoza cursului valutar EUR/RON
- bază de date SQLite care conține:
  - istoricul antrenărilor cu parametrii tunați și versionare:
    - model
    - listă parametri în format JSON
    - eroare medie / metrici înregistrate la antrenare
    - versiune model
  - datele istorice cu cursul valutar EUR/RON
  - prognoze istorice cu cel mai bun model
- backendul să facă scraping la cerere și să actualizeze datele istorice

# Rol

- ești freelancer senior de Python
- vei face design de UX
- vei face design de bază de date:
  - tabele
  - legături între ele
  - cod pentru menținerea bazei de date

# Cerințe

## Front end

- landing page cu prognoza curentă a cursului EUR/RON
- istoric curs valutar
- KPI care măsoară eroarea medie pe ultimele 14 zile
- indicator dacă eroarea are trend crescător sau descrescător
- un tab cu rezumatul antrenării și graficul existent cu backtest pe 14 zile
- un tab cu linkuri către studiile Optuna care să se deschidă la cerere într-un browser nou

## Back end

- endpoint pentru scraping la cerere
- endpoint pentru actualizarea datelor istorice
- endpoint pentru rularea antrenării rapide
- endpoint pentru pornirea optimizării Optuna
- endpoint pentru citirea ultimelor metrici
- endpoint pentru citirea prognozei curente
- endpoint pentru istoricul cursului

# Module de utilizat

- front end cu Streamlit
- back end cu FastAPI
- bază de date SQLite
