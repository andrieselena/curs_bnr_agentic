# Plan de Implementare: Optimizarea Hiperparametrilor

## Obiectiv
Îmbunătățirea acurateței predicțiilor prin ajustarea parametrilor interni (tuning) pentru modelele de Machine Learning, propunând și integrând ambele variante solicitate (GridSearch și Optuna), adaptate specificului seriilor de timp.

## Varianta 1: Implementarea cu GridSearch
Această variantă explorează exhaustiv o grilă predefinită de valori, testând mecanic fiecare combinație. Deoarece consumul de timp crește masiv, o vom direcționa spre modelul cu spațiu discret de parametri.

*   **Modele vizate:** SARIMA
*   **Hiperparametri optimizați:**
    *   `order`: combinații pentru (p, d, q) (ex: elementele să ia valori în intervalul [0, 1, 2]).
    *   `seasonal_order`: combinații pentru (P, D, Q, s) menținând sezonalitatea `s=5`.
*   **Metrica optimizată:** Root Mean Squared Error (RMSE) (obiectiv: minimizare).
*   **Validarea pentru serii temporale:** Deoarece datele au dependență temporală, cross-validarea clasică (K-Fold aleator) ar antrena pe date din viitor. Vom utiliza o tehnică de tip **Time Series Split / Rolling Forecast Origin**. Setul de antrenament va crește secvențial, iar validarea se va face strict pe datele din perioada imediat următoare pentru fiecare combinație de parametri.

## Varianta 2: Implementarea cu Optuna
Această variantă folosește optimizarea bayesiană (TPE) pentru a ghida inteligent căutarea în spații de parametri vaste, fiind ideală pentru modelele complexe care au hiperparametri cu valori continue.

*   **Modele vizate:** XGBoost Regressor
*   **Hiperparametri optimizați:**
    *   `learning_rate` (log-uniform, ex: 0.01 - 0.3)
    *   `n_estimators` (int, ex: 50 - 500)
    *   `max_depth` (int, ex: 3 - 9)
    *   `subsample` și `colsample_bytree` (uniform, ex: 0.6 - 1.0)
*   **Metrica optimizată:** Root Mean Squared Error (RMSE).
*   **Validarea pentru serii temporale:** Se va folosi nativ obiectul `TimeSeriesSplit` din `sklearn.model_selection`. Funcția `objective` din Optuna va antrena XGBoost-ul pe fiecare subset (fold) din TimeSeriesSplit, calculând RMSE. Media RMSE-urilor de pe toate foldurile de validare va reprezenta scorul combinației de parametri returnat către Optuna.

## Comparație și Recomandare
*   **GridSearch** este ideal când spațiul de parametri este mic și finit (cum este cazul valorilor p,d,q din SARIMA). Este 100% sigur, dar rigid.
*   **Optuna** este mult mai eficient computațional pentru spații imense și continue (XGBoost), având avantajul de a scurta execuțiile combinațiilor ineficiente (*pruning*).

**Recomandare:** Pentru a îndeplini cerința temei de a avea "o variantă cu..." pentru ambele abordări, recomand integrarea **simultană** a ambelor strategii. Concret: 
1. Vom crea o rutină de optimizare de tip **GridSearch manual/Time-Series** pentru a extrage optimul la **SARIMA**.
2. Vom crea un studiu avansat **Optuna** pentru a rafina hiperparametrii **XGBoost**.
*(Prophet va păstra parametrii de bază fiind folosit drept baseline robust)*.

## User Review Required
> [!IMPORTANT]
> Te rog să confirmi dacă ești de acord cu acest plan revizuit în care GridSearch și Optuna sunt folosite complementar pe modele diferite, fiecare implementând `TimeSeriesSplit` pentru a preveni `data leakage`.
> Dacă aprobi, vom decide exact în ce script implementăm codul (probabil un nou `tuning.py` pentru a menține separarea arhitecturală curată).
