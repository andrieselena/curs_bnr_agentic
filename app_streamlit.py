import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

# ==========================================
# CONFIGURAȚII ȘI CONSTANTE
# ==========================================
API_URL = "http://127.0.0.1:7772"
HTML_REPORT_PATH = Path("outputs/reports/grafic_prognoza_final.html")

st.set_page_config(page_title="BNR EUR/RON Full Stack", layout="wide")

# ==========================================
# FUNCȚII API CALLS
# ==========================================
def fetch_rates() -> pd.DataFrame:
    """Obține istoricul cursului valutar de la backend și returnează un DataFrame."""
    try:
        response = requests.get(f"{API_URL}/api/rates")
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
    except Exception as e:
        st.error(f"Eroare la conectarea cu API-ul (rates): {e}")
    return pd.DataFrame()

def fetch_metrics() -> dict:
    """Obține ultimele metrici de la backend."""
    try:
        response = requests.get(f"{API_URL}/api/metrics/latest")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"message": f"Eroare rețea sau backend: {e}"}

def trigger_scrape() -> dict:
    """Apelează endpoint-ul de scraping."""
    try:
        response = requests.post(f"{API_URL}/api/scrape")
        return response.json()
    except Exception as e:
        return {"message": f"Eroare API: {e}"}

def trigger_train() -> dict:
    """Apelează endpoint-ul de antrenare."""
    try:
        response = requests.post(f"{API_URL}/api/train")
        return response.json()
    except Exception as e:
        return {"message": f"Eroare API: {e}"}

# ==========================================
# INTERFAȚA WEB
# ==========================================
def main() -> None:
    """Entrypoint-ul pentru aplicația Streamlit Frontend."""
    st.title("Aplicație full stack curs BNR - EUR/RON")
    
    tab_landing, tab_train, tab_optuna = st.tabs(["Landing", "Antrenare", "Optuna"])
    
    # ---------------- TAB 1: LANDING ----------------
    with tab_landing:
        st.header("Prognoză curs EUR/RON")
        
        if st.button("Actualizează date", type="primary"):
            with st.spinner("Se extrag datele de la BNR. Vă rugăm așteptați..."):
                res = trigger_scrape()
                if "Eroare" in res.get("message", ""):
                    st.error(res.get("message"))
                else:
                    st.success(res.get("message", "Actualizare completă."))
        
        # Secțiune KPIs
        metrics = fetch_metrics()
        has_metrics = "mae" in metrics and metrics["mae"] is not None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if has_metrics:
                st.metric("MAE Curent", f"{metrics['mae']:.4f}")
            else:
                st.metric("MAE Curent", "Fără date")
                
        with col2:
            if has_metrics:
                st.metric("Trend Eroare", "Disponibil la 2+ rulări")
            else:
                st.metric("Trend Eroare", "Fără date")
            
        with col3:
            if has_metrics:
                st.metric("Model Câștigător", metrics.get("model_name", "N/A"))
            else:
                st.metric("Model Câștigător", "Fără date")
        
        if not has_metrics and "message" in metrics:
            st.info(metrics["message"])
        
        # Grafic Istoric
        st.subheader("Istoric Curs EUR/RON")
        df_rates = fetch_rates()
        if not df_rates.empty:
            st.line_chart(df_rates['value'], use_container_width=True)
        else:
            st.info("Nu există date istorice în baza de date. Apăsați butonul 'Actualizează date' sau porniți serverul Backend.")

    # ---------------- TAB 2: ANTRENARE ----------------
    with tab_train:
        st.header("Rezumat Antrenare și Backtest")
        
        if st.button("Reantrenare rapidă"):
            with st.spinner("Se comunică cu Backend-ul..."):
                res_train = trigger_train()
                st.info(res_train.get("message", ""))
                
        st.subheader("Ultimele metrici înregistrate")
        if has_metrics:
            st.json(metrics)
        else:
            st.info("Nicio antrenare înregistrată recent în baza de date SQLite.")
            
        st.subheader("Grafic de Backtest (Ultimele 14 zile)")
        if HTML_REPORT_PATH.exists():
            with open(HTML_REPORT_PATH, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=600, scrolling=True)
        else:
            st.info("Graficul HTML interactiv nu a fost generat încă. Rulați antrenarea din terminal (`python run_training_pipeline.py`).")

    # ---------------- TAB 3: OPTUNA ----------------
    with tab_optuna:
        st.header("Studiu Optimizare Hiperparametri (Optuna)")
        
        st.info("Pentru ca linkurile de mai jos să funcționeze, pornește dashboard-urile Optuna în terminale separate.")
        
        st.code(
            "optuna-dashboard sqlite:///outputs/optuna_studies/xgboost_optimization.db --port 8090\n"
            "optuna-dashboard sqlite:///outputs/optuna_studies/sarima_optimization.db --port 8091\n"
            "optuna-dashboard sqlite:///outputs/optuna_studies/prophet_optimization.db --port 8092",
            language="bash"
        )
        
        st.warning("⚠️ ATENȚIE: Dashboard-urile de mai jos NU funcționează automat! Trebuie să rulați comenzile de mai sus sau notebook-ul `pornire_dashboard_optuna.ipynb` din terminal/IDE pentru a porni serverele.")
        
        st.markdown(
            """
            **Legături către porturile locale:**
            - [XGBoost Dashboard](http://127.0.0.1:8090)
            - [SARIMA Dashboard](http://127.0.0.1:8091)
            - [Prophet Dashboard](http://127.0.0.1:8092)
            """
        )

if __name__ == "__main__":
    main()
