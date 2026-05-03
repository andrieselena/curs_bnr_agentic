# Plan de Implementare: Modele de Prognoză a Cursului de Schimb

Acest plan detaliază pașii necesari pentru prelucrarea datelor, antrenarea, evaluarea și testarea a 3 modele concurente menite să prognozeze cursul valutar pentru ziua următoare (Z+1), utilizând datele deja extrase (CSV-ul cu seriile de timp).

## 1. Pregătirea și Prelucrarea Datelor (Data Preprocessing)
- Încărcarea datelor istorice din CSV într-un dataframe.
- Curățarea datelor: conversia tipurilor de date, formatarea corectă a coloanei „Data” și setarea ei ca index, tratarea posibilelor valori lipsă (ex: interpolare pentru weekend-uri sau sărbători legale).
- **Împărțirea setului de date (Train / Test Split):**
  - **Test Set:** Ultimele 14 zile cronologice ale setului de date, destinate evaluării finale.
  - **Train Set:** Restul datelor (de la 22/02/2020 până la Test Set minus 1 zi) va fi folosit pentru antrenament.
- **Feature Engineering** (în special necesar pentru cel puțin al doilea model): generarea de coloane derivate, cum ar fi cursul pe zilele anterioare (lag-uri la t-1, t-2, t-3), zi din săptămână, lună și eventual medii mobile pe 7 sau 14 zile.

## 2. Selecția și Antrenarea celor 3 Modele
Se propun următoarele 3 topologii diferite, recunoscute în prognoza de tip Time Series:

1. **Modelul 1: Prophet (Meta)**
   - Dedicat seriilor financiare, scalează natural cu zilele lipsă (sărbători legale), modelează excelent sezonalitatea săptămânală sau anuală și oferă inerent intervale de încredere.
2. **Modelul 2: XGBoost Regressor**
   - Un model modern și foarte puternic de tip _Gradient Boosting_. Vom folosi ca date de intrare variabilele derivate (feature engineering-ul) - istoria pe N zile anterioare pentru a deduce următoarea zi.
3. **Modelul 3: ARIMA / SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
   - Standardul statistic, matematic și clasic robust pentru previzionarea dinamică a valorilor univariate luând în calcul proprietățile de autocorelare din cursul BNR.

## 3. Raportul de Evaluare (Evaluarea Performanței)
Vom prezenta un raport tabelar în care testăm cele 3 modele de mai sus pe același Test Set (ultimele 14 zile), calculând următoarele metrici de precizie relevante:
- **MAE** (*Mean Absolute Error*) - Exprimată direct în RON (Ex: modelul s-a înșelat în medie cu 0.0051 lei pe zi). 
- **RMSE** (*Root Mean Squared Error*) - Penalizează sever erorile mai mari, forțând o stabilitate generală mai mare.
- **MAPE** (*Mean Absolute Percentage Error*) - Procentual (Ex: cursul a fost prezis cu o diferență procentuală de 0.1%).

## 4. Salvarea Celui Mai Bun Model
- În baza raportului generat la Punctul 3, vom realiza o evaluare comparativă a metricelor (modelul ce obține valorile minime la MAE / RMSE este declarat "câștigător").
- **Persistență**: Vom salva local structura pre-antrenată a "Câștigătorului" utilizând funcțiile specifice (ex. `joblib`, `.pkl` sau fișier propriu prophet), pentru a putea fi ulterior încărcat doar pentru estimarea Z+1, fără a reitera procesul de învățare.

## 5. Vizualizarea Plotly (Forecasting Dashboards)
- Modelul câștigător va oferi o re-predicție pe cele 14 zile din test set. 
- Vom construi vizualizarea unui grafic dinamic web-ready folosind librăria **Plotly**, care va conține obligatoriu:
  - O linie care detaliază **Cursul Oficial / Adevărat** pe cele două săptămâni (Real Data).
  - O linie care detaliază estimările **(Prognoza)** modelului câștigător.
  - Generarea unei raze **(Confidence Interval)** superioare și inferioare (Ex. 95%) - de regulă redată ca un „shaded area” deasupra și sub linia cu valoarea estimată, demonstrând volatilitatea percepută.