import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components
from requests.exceptions import RequestException

try:
    from src.curs_bnr.model_utils import run_bnr_agent
except Exception:
    run_bnr_agent = None

# ==========================================
# CONFIGURAȚII ȘI CONSTANTE
# ==========================================
API_URL = "http://127.0.0.1:7772"
HTML_REPORT_PATH = Path("outputs/reports/grafic_prognoza_final.html")
OPTUNA_DASHBOARDS = {
    "XGBoost": "http://127.0.0.1:8090",
    "SARIMA": "http://127.0.0.1:8091",
    "Prophet": "http://127.0.0.1:8092",
}

st.set_page_config(page_title="Prognoză curs EUR/RON", layout="wide")

PAGE_STYLE = """
<style>
    * {font-family: 'Inter', sans-serif;}
    .page-title {font-size: 2.75rem; font-weight: 800; margin-bottom: 0.35rem;}
    .page-subtitle {font-size: 1rem; color: #556581; margin-bottom: 1.75rem; line-height: 1.6;}
    .card {background: #ffffff; border-radius: 20px; padding: 24px; box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08); margin-bottom: 20px;}
    .card-header {font-size: 0.95rem; color: #6b7280; margin-bottom: 0.65rem;}
    .card-value {font-size: 2.1rem; font-weight: 700; color: #111827; line-height: 1.1;}
    .section-title {font-size: 1.3rem; font-weight: 700; margin-bottom: 0.75rem;}
    .section-card {background: #f8fafc; border-radius: 18px; padding: 22px; margin-bottom: 24px;}
    .small-card {background: #ffffff; border-radius: 16px; padding: 18px; box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06); margin-bottom: 16px;}
    .button-link {display: inline-flex; align-items: center; justify-content: center; padding: 0.85rem 1.25rem; border-radius: 12px; background: #2563eb; color: white; text-decoration: none; font-weight: 600;}
    .button-link:hover {background: #1d4ed8;}
    .nav-button {width: 100%; margin-bottom: 0.75rem;}
    .sidebar-description {font-size: 0.95rem; color: #6b7280; margin-bottom: 1rem;}
    .chat-example {background: #f3f4f6; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; color: #0f172a;}
    .chat-example strong {display: block; margin-bottom: 0.25rem;}
    .link-label {font-size: 0.95rem; color: #2563eb;}
    .section-note {font-size: 0.95rem; color: #475569; margin-bottom: 16px;}
</style>
"""

st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# ==========================================
# FUNCȚII API CALLS
# ==========================================

def backend_error(detail: str | None = None) -> dict:
    message = "Backend-ul nu este pornit. Rulează: python run_backend.py"
    if detail:
        message += f" | {detail}"
    return {"error": message}


def safe_get(path: str, timeout: int = 15) -> dict | list:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except RequestException as exc:
        return backend_error(str(exc))


def safe_post(path: str, timeout: int = 30, payload: dict | None = None) -> dict:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except RequestException as exc:
        return backend_error(str(exc))


def is_backend_down(result: object) -> bool:
    return isinstance(result, dict) and "error" in result and str(result["error"]).startswith("Backend-ul nu este pornit")


def get_historical_rates() -> pd.DataFrame:
    result = safe_get("/api/rates")
    if isinstance(result, list) and result:
        df = pd.DataFrame(result)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        return df
    return pd.DataFrame()

def get_latest_metrics() -> dict:
    result = safe_get("/api/metrics/latest")
    if isinstance(result, dict):
        return result
    return {"message": "Răspuns neașteptat de la backend."}


def get_forecast_summary() -> dict:
    result = safe_get("/api/forecast/latest")
    if is_backend_down(result):
        return result
    if isinstance(result, list):
        return {"count": len(result), "data": result}
    if isinstance(result, dict) and result.get("message"):
        return {"count": 0, "message": result.get("message"), "data": []}
    return {"count": 0, "data": []}


def trigger_train() -> dict:
    return safe_post("/api/train", timeout=30)


def get_training_runs() -> dict:
    result = safe_get("/api/runs")
    if is_backend_down(result):
        return result
    if isinstance(result, list):
        return {"success": True, "runs": result}
    return {"success": False, "message": "Răspuns invalid pentru rulări."}


def get_optuna_info() -> dict:
    result = safe_post("/api/optuna", timeout=15)
    dashboards = OPTUNA_DASHBOARDS
    if isinstance(result, dict) and result.get("dashboards"):
        return {
            "success": True,
            "dashboards": dashboards,
            "message": result.get("message", "Informații Optuna încărcate."),
            "backend_dashboards": result.get("dashboards"),
        }
    if is_backend_down(result):
        return result
    return {"success": True, "dashboards": dashboards, "message": "Folosesc porturile corecte Optuna locale."}


def format_currency(value: float | int | str) -> str:
    try:
        return f"{float(value):,.4f}"
    except Exception:
        return str(value)


def render_card(title: str, value: str, help_text: str | None = None) -> None:
    st.markdown(
        f"""
        <div class='card'>
            <div class='card-header'>{title}</div>
            <div class='card-value'>{value}</div>
            {f"<div style='color:#6b7280;margin-top:0.75rem;font-size:0.95rem;'>{help_text}</div>" if help_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_home() -> None:
    st.markdown("<div class='page-title'>Prognoză curs EUR/RON</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Explorează evoluția euro în raport cu leul și urmărește cele mai recente rezultate ale modelelor.</div>",
        unsafe_allow_html=True,
    )

    rates = get_historical_rates()
    metrics = get_latest_metrics()
    forecast = get_forecast_summary()
    optimal = get_optuna_info()

    latest_rate = "N/A"
    if not rates.empty:
        latest_rate = format_currency(rates["value"].iloc[-1])

    model_name = metrics.get("model_name") or metrics.get("model") or metrics.get("message", "N/A")
    mae_value = format_currency(metrics.get("mae", "N/A"))
    forecast_count = forecast.get("count", 0)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="large")
    with kpi1:
        render_card("Ultimul curs disponibil", f"{latest_rate} EUR/RON", "Ultima valoare observată în baza de date.")
    with kpi2:
        render_card("Model câștigător", str(model_name), "Modelul evaluat cu cea mai mică eroare.")
    with kpi3:
        render_card("MAE curent", mae_value, "Eroarea medie absolută a predicțiilor.")
    with kpi4:
        render_card("Prognoze disponibile", str(forecast_count), "Numărul de valori de prognoză stocate.")

    if is_backend_down(metrics) or is_backend_down(forecast) or is_backend_down(optimal):
        st.error("Backend-ul nu este pornit. Rulează: python run_backend.py")

    st.markdown("<div class='section-title'>Grafic istoric</div>", unsafe_allow_html=True)
    if not rates.empty:
        plot_data = rates.copy().reset_index()
        plot_data["date"] = pd.to_datetime(plot_data["date"], errors="coerce")
        plot_data["value"] = pd.to_numeric(plot_data["value"], errors="coerce")
        plot_data = plot_data.dropna(subset=["date", "value"]).sort_values(by="date")

        if not plot_data.empty:
            y_min = float(plot_data["value"].min()) - 0.02
            y_max = float(plot_data["value"].max()) + 0.02
            if y_max - y_min < 0.05:
                center = (y_max + y_min) / 2
                y_min = center - 0.025
                y_max = center + 0.025

            fig = px.line(
                plot_data,
                x="date",
                y="value",
                title="Istoric curs EUR/RON",
                labels={"date": "Data", "value": "Curs EUR/RON"},
                template="plotly_white",
                markers=False,
            )
            fig.update_traces(
                mode="lines",
                hovertemplate="%{x|%d %b %Y}<br>Curs EUR/RON: %{y:.4f}",
            )
            fig.update_layout(
                yaxis=dict(range=[y_min, y_max]),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nu există date istorice disponibile. Rulează scraping-ul.")
    else:
        st.info("Nu există date istorice disponibile. Rulează scraping-ul.")

    st.markdown("<div class='section-title'>Grafic prognoză - model câștigător</div>", unsafe_allow_html=True)
    if HTML_REPORT_PATH.exists():
        with open(HTML_REPORT_PATH, "r", encoding="utf-8") as html_file:
            components.html(html_file.read(), height=700, scrolling=True)
    else:
        st.info("Graficul nu a fost generat încă. Rulează: python run_training_pipeline.py")

    st.markdown("---")
    st.markdown("<div class='section-title'>Navigare rapidă</div>", unsafe_allow_html=True)
    button1, button2, button3 = st.columns(3, gap="large")
    with button1:
        if st.button("🧠 Antrenare", key="home_train"):
            st.session_state.page = "🧠 Antrenare"
            st.experimental_rerun()
    with button2:
        if st.button("📊 Optuna Dashboards", key="home_optuna"):
            st.session_state.page = "📊 Optuna Dashboards"
            st.experimental_rerun()
    with button3:
        if st.button("💬 Chatbot", key="home_chatbot"):
            st.session_state.page = "💬 Chatbot"
            st.experimental_rerun()

    if not is_backend_down(optimal) and isinstance(optimal, dict) and optimal.get("success"):
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.write(optimal.get("message"))
        st.markdown("<strong>Porturi corecte folosite în aplicație:</strong>", unsafe_allow_html=True)
        for name, url in OPTUNA_DASHBOARDS.items():
            st.markdown(f"- **{name}**: {url}")
        st.markdown("</div>", unsafe_allow_html=True)


def page_training() -> None:
    st.markdown("<div class='page-title'>Antrenare și model câștigător</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Modelele SARIMA, Prophet și XGBoost sunt evaluate pe ultimele 14 zile, iar aplicația salvează modelul cu cea mai mică eroare.</div>",
        unsafe_allow_html=True,
    )

    runs = get_training_runs()
    metrics = get_latest_metrics()

    if is_backend_down(runs) or is_backend_down(metrics):
        st.error("Backend-ul nu este pornit. Rulează: python run_backend.py")
        return

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        render_card(
            "Model câștigător",
            str(metrics.get("model_name") or metrics.get("model") or "N/A"),
            "Modelul selectat pentru cea mai bună performanță.",
        )
    with col2:
        render_card("MAE", format_currency(metrics.get("mae", "N/A")), "Eroarea medie absolută a predicțiilor.")
    with col3:
        render_card("RMSE", format_currency(metrics.get("rmse", "N/A")), "Rădăcina medie a erorii pătratice.")
    with col4:
        render_card("MAPE", format_currency(metrics.get("mape", "N/A")), "Eroarea procentuală medie absolută.")

    sub1, sub2 = st.columns(2, gap="large")
    with sub1:
        render_card("Versiune", str(metrics.get("version", "N/A")), "Versiunea ultimului model.")
    with sub2:
        render_card("Timestamp", str(metrics.get("timestamp", "N/A")), "Momentul ultimei evaluări.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ultima rulare</div>", unsafe_allow_html=True)
    if runs.get("success") and runs.get("runs"):
        latest_run = runs["runs"][-1] if isinstance(runs["runs"], list) and runs["runs"] else {}
        run_date = latest_run.get("timestamp") or latest_run.get("date") or latest_run.get("run_date") or "N/A"
        run_model = latest_run.get("model_name") or latest_run.get("model") or "N/A"
        run_version = latest_run.get("version") or "N/A"
        hyperparams = latest_run.get("hyperparameters") or latest_run.get("params") or latest_run.get("hparams") or {}

        run_col1, run_col2, run_col3 = st.columns(3, gap="large")
        with run_col1:
            render_card("Data rulării", str(run_date), "Momentul ultimei execuții din pipeline.")
        with run_col2:
            render_card("Model", str(run_model), "Modelul folosit în ultima rulare.")
        with run_col3:
            render_card("Versiune", str(run_version), "Versiunea modelului din ultima rulare.")

        with st.expander("Hiperparametri", expanded=False):
            if isinstance(hyperparams, dict) and hyperparams:
                for key, value in hyperparams.items():
                    st.write(f"- **{key}**: {value}")
            elif hyperparams:
                st.write(str(hyperparams))
            else:
                st.write("Niciun hiperparametru disponibil.")
    else:
        st.info(runs.get("message", "Nu există informații despre ultima rulare."))

    st.markdown("---")
    st.markdown("<div class='section-title'>Grafic prognoză - model câștigător</div>", unsafe_allow_html=True)
    if HTML_REPORT_PATH.exists():
        with open(HTML_REPORT_PATH, "r", encoding="utf-8") as html_file:
            components.html(html_file.read(), height=700, scrolling=True)
    else:
        st.info("Graficul nu a fost generat încă. Rulează: python run_training_pipeline.py")

    st.markdown("---")
    st.markdown(
        "<div class='card'>"
        "<div class='card-header'>Reantrenare</div>"
        "<div class='card-value' style='font-size:1rem; font-weight:500;'>" 
        "Pentru siguranță, reantrenarea completă nu se rulează direct din Streamlit." 
        "</div>"
        "<div style='color:#6b7280;margin-top:0.75rem;font-size:0.95rem;'>" 
        "Streamlit poate solicita doar informații de status și feedback de la backend."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Verifică status reantrenare"):
        result = trigger_train()
        if is_backend_down(result):
            st.error(result["error"])
        else:
            st.success(result.get("message", "A fost solicitată informația de antrenare."))

    st.code("python run_training_pipeline.py", language="bash")

    with st.expander("Detalii tehnice pentru debug", expanded=False):
        st.write("Metrici brute și informații despre rulări.")
        st.json(metrics)
        st.json(runs)


def page_optuna() -> None:
    st.markdown("<div class='page-title'>Optuna Dashboards</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Valoarea Objective reprezintă eroarea; mai mic este mai bun.</div>",
        unsafe_allow_html=True,
    )

    info = get_optuna_info()
    if is_backend_down(info):
        st.error(info["error"])

    st.markdown(
        "<div class='section-note'>Folosește dashboard-urile locale pentru a compara vizual performanța XGBoost, SARIMA și Prophet.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="large")
    model_details = [
        ("XGBoost", "http://127.0.0.1:8090", "optuna-dashboard sqlite:///outputs/optuna_studies/xgboost_optimization.db --port 8090", "8090"),
        ("SARIMA", "http://127.0.0.1:8091", "optuna-dashboard sqlite:///outputs/optuna_studies/sarima_optimization.db --port 8091", "8091"),
        ("Prophet", "http://127.0.0.1:8092", "optuna-dashboard sqlite:///outputs/optuna_studies/prophet_optimization.db --port 8092", "8092"),
    ]
    for index, (name, url, command, port) in enumerate(model_details):
        with cols[index]:
            st.markdown(
                f"""
                <div class='card'>
                    <div class='card-header'>{name}</div>
                    <div class='card-value'>{url}</div>
                    <div style='margin-top:0.75rem; color:#6b7280;'>Port local corect: {port}</div>
                    <div style='margin-top:0.85rem;'>
                        <a class='button-link' href='{url}' target='_blank'>Deschide dashboard</a>
                    </div>
                    <div style='margin-top:0.85rem; font-size:0.95rem; color:#334155;'>Comandă de pornire:</div>
                    <div class='code-card'><code>{command}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if isinstance(info, dict) and info.get("backend_dashboards"):
        st.markdown("**Backend-ul FastAPI a răspuns cu următoarele dashboard-uri:**")
        for model_name, backend_url in info.get("backend_dashboards", {}).items():
            st.write(f"- {model_name}: {backend_url}")
        st.info("Frontend-ul folosește porturile corecte 8090/8091/8092, chiar dacă backend-ul returnează porturi vechi.")


def page_chatbot() -> None:
    st.markdown("<div class='page-title'>Chatbot EUR/RON</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Conversează cu agentul pentru a obține informații rapide despre date, modele și Optuna.</div>",
        unsafe_allow_html=True,
    )

    if run_bnr_agent is None:
        st.error(
            "Importul `run_bnr_agent` a eșuat. Asigură-te că rulezi aplicația din directorul proiectului și că `src` este accesibil."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.markdown("<div class='section-title'>Întrebări recomandate</div>", unsafe_allow_html=True)
    example1, example2, example3 = st.columns(3, gap="large")
    with example1:
        st.markdown("<div class='small-card'><strong>Care este ultimul curs EUR/RON?</strong></div>", unsafe_allow_html=True)
    with example2:
        st.markdown("<div class='small-card'><strong>Ce model este câștigător?</strong></div>", unsafe_allow_html=True)
    with example3:
        st.markdown("<div class='small-card'><strong>Cum pornesc dashboard-urile Optuna?</strong></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<strong>Zona de chat</strong>", unsafe_allow_html=True)
    user_message = st.chat_input("Scrie întrebarea ta aici...")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        try:
            answer = run_bnr_agent(user_message, verbose=False)
        except Exception as exc:
            answer = f"Eroare la rularea chatbotului: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 Acasă / Prognoză"

    st.sidebar.title("Navigare")
    st.sidebar.markdown("<div class='sidebar-description'>Explorează rapid aplicația și găsește secțiunea dorită.</div>", unsafe_allow_html=True)
    page_options = [
        "🏠 Acasă / Prognoză",
        "🧠 Antrenare",
        "📊 Optuna Dashboards",
        "💬 Chatbot",
    ]
    page = st.sidebar.radio(
        "",
        page_options,
        index=page_options.index(st.session_state["page"] if st.session_state["page"] in page_options else page_options[0]),
    )
    st.session_state["page"] = page

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Aplicație EUR/RON Full Stack cu pagini dedicate pentru prognoză, antrenare, Optuna și chatbot."
    )

    if page == "🏠 Acasă / Prognoză":
        page_home()
    elif page == "🧠 Antrenare":
        page_training()
    elif page == "📊 Optuna Dashboards":
        page_optuna()
    elif page == "💬 Chatbot":
        page_chatbot()


if __name__ == "__main__":
    main()
