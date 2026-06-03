"""
model_utils.py

Agent Gemini simplu cu tool calling local pentru analiza cursului EUR/RON.
Tool-urile sunt funcții Python locale care apelează backend-ul FastAPI existent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import requests
from dotenv import load_dotenv

import google.generativeai as genai


# =============================================================================
# CONFIGURARE
# =============================================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
BNR_BACKEND_URL = os.getenv("BNR_BACKEND_URL", "http://127.0.0.1:7772")

LOCAL_OPTUNA_DASHBOARDS: dict[str, str] = {
    "XGBoost": "http://127.0.0.1:8090",
    "SARIMA": "http://127.0.0.1:8091",
    "Prophet": "http://127.0.0.1:8092",
}

SYSTEM_INSTRUCTION = (
    "Ești un asistent specializat în analiza cursului valutar EUR/RON. "
    "Răspunzi în limba română, clar și concis. "
    "Folosești rezultatele tool-urilor locale atunci când acestea sunt disponibile. "
    "Nu inventezi valori. Dacă lipsesc date, explici ce trebuie rulat."
)


# =============================================================================
# FUNCȚII AUXILIARE
# =============================================================================

def _parse_json_response(response: requests.Response) -> Any:
    """Parsează răspunsul JSON sau returnează text brut dacă răspunsul nu este JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text


def _safe_get(url: str, timeout: int = 15) -> Any:
    """Execută un request GET și returnează JSON/text sau eroare controlată."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return _parse_json_response(response)
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": "Nu s-a putut apela backend-ul.",
            "details": str(exc),
        }


def _safe_post(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    """Execută un request POST și returnează JSON/text sau eroare controlată."""
    try:
        if payload is None:
            response = requests.post(url, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)

        response.raise_for_status()
        return _parse_json_response(response)
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": "Nu s-a putut apela backend-ul.",
            "details": str(exc),
        }


# =============================================================================
# TOOL-URI LOCALE
# =============================================================================

def get_latest_exchange_rate() -> dict[str, Any]:
    """Returnează ultimul curs EUR/RON disponibil din backend."""
    data = _safe_get(f"{BNR_BACKEND_URL}/api/rates")

    if isinstance(data, dict) and data.get("success") is False:
        return data

    if not isinstance(data, list) or not data:
        return {
            "success": False,
            "message": "Nu există date de curs disponibile. Rulează scraping-ul sau pipeline-ul.",
        }

    latest_rate = data[-1]

    return {
        "success": True,
        "currency": "EUR/RON",
        "date": latest_rate.get("date"),
        "value": latest_rate.get("value"),
        "message": (
            f"Ultimul curs EUR/RON disponibil este {latest_rate.get('value')} "
            f"la data {latest_rate.get('date')}."
        ),
    }


def get_forecast_summary() -> dict[str, Any]:
    """Returnează un sumar al ultimei prognoze EUR/RON disponibile."""
    data = _safe_get(f"{BNR_BACKEND_URL}/api/forecast/latest")

    if isinstance(data, dict) and data.get("success") is False:
        return data

    if isinstance(data, dict) and data.get("message"):
        return {
            "success": False,
            "message": data.get("message"),
        }

    forecasts: list[dict[str, Any]] = []

    if isinstance(data, list):
        forecasts = data
    elif isinstance(data, dict):
        if isinstance(data.get("forecasts"), list):
            forecasts = data["forecasts"]
        elif isinstance(data.get("forecast"), list):
            forecasts = data["forecast"]
        elif isinstance(data.get("data"), list):
            forecasts = data["data"]
        elif data.get("forecast_date") or data.get("forecast_value"):
            forecasts = [data]

    if not forecasts:
        return {
            "success": False,
            "message": "Nu există prognoză salvată. Rulează: python run_training_pipeline.py",
        }

    preview = forecasts[:5]
    last_item = forecasts[-1]

    return {
        "success": True,
        "currency": "EUR/RON",
        "forecast_count": len(forecasts),
        "preview": preview,
        "latest_forecast_date": (
            last_item.get("forecast_date")
            or last_item.get("date")
            or last_item.get("ds")
        ),
        "latest_forecast_value": (
            last_item.get("forecast_value")
            or last_item.get("predicted_value")
            or last_item.get("prediction")
            or last_item.get("value")
        ),
        "message": "Am găsit ultima prognoză EUR/RON disponibilă.",
    }


def get_model_metrics() -> dict[str, Any]:
    """Returnează modelul câștigător și ultimele metrici MAE, RMSE, MAPE."""
    data = _safe_get(f"{BNR_BACKEND_URL}/api/metrics/latest")

    if isinstance(data, dict) and data.get("success") is False:
        return data

    if isinstance(data, dict) and data.get("message"):
        return {
            "success": False,
            "message": data.get("message"),
        }

    if not isinstance(data, dict):
        return {
            "success": False,
            "message": "Răspuns invalid de la backend pentru metrici.",
        }

    model_name = (
        data.get("model_name")
        or data.get("winner_model")
        or data.get("model")
        or data.get("name")
    )

    return {
        "success": True,
        "currency": "EUR/RON",
        "model_name": model_name,
        "mae": data.get("mae"),
        "rmse": data.get("rmse"),
        "mape": data.get("mape"),
        "version": data.get("version"),
        "timestamp": data.get("timestamp") or data.get("run_at"),
        "message": f"Modelul câștigător este {model_name}.",
    }


def get_training_runs() -> dict[str, Any]:
    """Returnează ultimele rulări de antrenare din backend."""
    data = _safe_get(f"{BNR_BACKEND_URL}/api/runs")

    if isinstance(data, dict) and data.get("success") is False:
        return data

    if not isinstance(data, list):
        return {
            "success": False,
            "message": "Răspuns invalid de la backend pentru rulări.",
        }

    if not data:
        return {
            "success": False,
            "message": "Nu există rulări de antrenare. Rulează: python run_training_pipeline.py",
        }

    return {
        "success": True,
        "training_runs_count": len(data),
        "latest_runs": data[:5],
        "message": f"Există {len(data)} rulări de antrenare în backend.",
    }


def trigger_scrape() -> dict[str, Any]:
    """
    Pornește scraping-ul prin backend.

    În proiectul curent endpoint-ul /api/scrape este deja configurat pentru EUR/RON,
    deci încercăm întâi fără body. Dacă backend-ul cere body, încercăm fallback cu EUR.
    """
    url = f"{BNR_BACKEND_URL}/api/scrape"

    result = _safe_post(url, payload=None, timeout=60)

    if isinstance(result, dict) and result.get("success") is False:
        fallback = _safe_post(url, payload={"currency_code": "EUR"}, timeout=60)
        return {
            "success": not (isinstance(fallback, dict) and fallback.get("success") is False),
            "currency": "EUR/RON",
            "data": fallback,
            "message": "Am încercat actualizarea datelor EUR/RON prin scraping.",
        }

    return {
        "success": True,
        "currency": "EUR/RON",
        "data": result,
        "message": "Actualizarea datelor EUR/RON a fost pornită.",
    }


def get_optuna_info() -> dict[str, Any]:
    """Returnează informații despre dashboard-urile Optuna."""
    data = _safe_post(f"{BNR_BACKEND_URL}/api/optuna", payload=None, timeout=15)

    if isinstance(data, dict) and data.get("dashboards"):
        return {
            "success": True,
            "dashboards": data.get("dashboards"),
            "message": "Am găsit dashboard-urile Optuna în backend.",
        }

    return {
        "success": True,
        "dashboards": LOCAL_OPTUNA_DASHBOARDS,
        "message": "Folosesc linkurile locale implicite pentru dashboard-urile Optuna.",
    }


# =============================================================================
# REGISTRU TOOL-URI
# =============================================================================

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_latest_exchange_rate": get_latest_exchange_rate,
    "get_forecast_summary": get_forecast_summary,
    "get_model_metrics": get_model_metrics,
    "get_training_runs": get_training_runs,
    "trigger_scrape": trigger_scrape,
    "get_optuna_info": get_optuna_info,
}


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "get_latest_exchange_rate",
        "description": "Returnează ultimul curs EUR/RON disponibil.",
    },
    {
        "name": "get_forecast_summary",
        "description": "Returnează ultima prognoză EUR/RON disponibilă.",
    },
    {
        "name": "get_model_metrics",
        "description": "Returnează modelul câștigător și metricile MAE, RMSE, MAPE.",
    },
    {
        "name": "get_training_runs",
        "description": "Returnează ultimele rulări de antrenare.",
    },
    {
        "name": "trigger_scrape",
        "description": "Pornește actualizarea datelor EUR/RON prin scraping.",
    },
    {
        "name": "get_optuna_info",
        "description": "Returnează informații despre dashboard-urile Optuna.",
    },
]


# =============================================================================
# TOOL ROUTING LOCAL
# =============================================================================

def choose_tool_for_message(user_message: str) -> tuple[str | None, dict[str, Any]]:
    """
    Alege local tool-ul potrivit pentru mesajul utilizatorului.

    Nu se bazează pe function calling nativ Gemini, pentru a evita erorile de schemă.
    """
    message = user_message.lower()

    if any(word in message for word in ["optuna", "dashboard"]):
        return "get_optuna_info", {}

    if any(word in message for word in ["actualizeaza", "actualizează", "scrape", "scraping", "date noi"]):
        return "trigger_scrape", {}

    if any(word in message for word in ["prognoza", "prognoză", "predictie", "predicție", "forecast"]):
        return "get_forecast_summary", {}

    if any(word in message for word in ["mae", "rmse", "mape", "eroare", "castigator", "câștigător", "model"]):
        return "get_model_metrics", {}

    if any(word in message for word in ["rulare", "rulări", "rulari", "antrenare", "training", "antrenări"]):
        return "get_training_runs", {}

    if any(word in message for word in ["curs", "valoare", "istoric"]):
        return "get_latest_exchange_rate", {}

    return None, {}


def execute_tool_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Execută un tool local și returnează rezultatul ca JSON string."""
    func = TOOL_REGISTRY.get(tool_name)

    if func is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Tool necunoscut: {tool_name}",
            },
            ensure_ascii=False,
        )

    try:
        result = func(**tool_args)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Argumente invalide pentru {tool_name}: {exc}",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Eroare la execuția {tool_name}: {exc}",
            },
            ensure_ascii=False,
        )


# =============================================================================
# GEMINI AGENT
# =============================================================================

def _get_gemini_model() -> genai.GenerativeModel:
    """Configurează și returnează modelul Gemini."""
    if not GEMINI_API_KEY:
        raise ValueError("Setează GEMINI_API_KEY sau GOOGLE_API_KEY în fișierul .env.")

    genai.configure(api_key=GEMINI_API_KEY)

    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )


def run_bnr_agent(user_message: str, verbose: bool = True) -> str:
    """
    Rulează agentul EUR/RON.

    Flux:
    1. Alege local tool-ul potrivit.
    2. Execută tool-ul.
    3. Trimite rezultatul tool-ului către Gemini.
    4. Gemini formulează răspunsul final în română.
    """
    tool_name, tool_args = choose_tool_for_message(user_message)

    if verbose:
        print(f"[USER] {user_message}")
        print(f"[TOOL_SELECTED] {tool_name}")

    tool_result_json = None

    if tool_name:
        tool_result_json = execute_tool_call(tool_name, tool_args)

        if verbose:
            print(f"[TOOL_RESULT] {tool_result_json}")

    model = _get_gemini_model()

    if tool_result_json:
        prompt = f"""
Întrebarea utilizatorului:
{user_message}

Tool apelat local:
{tool_name}

Rezultatul tool-ului local, în format JSON:
{tool_result_json}

Formulează un răspuns scurt, clar și natural, în limba română.
Nu inventa valori.
Dacă rezultatul conține o eroare sau lipsă de date, explică exact ce trebuie rulat.
"""
    else:
        prompt = f"""
{SYSTEM_INSTRUCTION}

Întrebarea utilizatorului:
{user_message}

Nu există un tool local potrivit pentru această întrebare.
Răspunde general, în română, fără să inventezi valori numerice.
"""

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        return f"Eroare la apelarea Gemini: {exc}"

    if not response or not getattr(response, "text", None):
        return "Nu am primit răspuns de la Gemini."

    final_text = response.text.strip()

    if verbose:
        print(f"[AGENT] {final_text}")

    return final_text


# =============================================================================
# DEMO CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("BNR EUR/RON Agent Gemini - Demo")
    print("=" * 70)
    print(f"Model Gemini: {GEMINI_MODEL}")
    print(f"Backend: {BNR_BACKEND_URL}")
    print()

    demo_questions = [
        "Care este ultimul curs EUR/RON disponibil?",
        "Ce model este câștigător și ce eroare are?",
        "Arată-mi ultima prognoză disponibilă.",
        "Cum pornesc dashboard-urile Optuna?",
    ]

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = run_bnr_agent(question, verbose=True)
        print()
        print("Răspuns final:")
        print(answer)
    else:
        for index, question in enumerate(demo_questions, start=1):
            print(f"Demo {index}/{len(demo_questions)}: {question}")
            print("-" * 70)

            selected_tool, _ = choose_tool_for_message(question)
            print(f"Tool ales local: {selected_tool}")

            try:
                answer = run_bnr_agent(question, verbose=False)
                print(f"Răspuns: {answer}")
            except Exception as exc:
                print(f"Eroare: {exc}")

            print()