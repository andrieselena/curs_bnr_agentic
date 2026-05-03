import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from src.curs_bnr.config import CURRENCY


def create_forecast_plot(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    model_name: str = "Model Câștigător",
    currency: str = CURRENCY,
    save_html_path: Optional[str] = "grafic_prognoza.html"
) -> go.Figure:
    """
    Construiește și (opțional) salvează un grafic interactiv Plotly.

    Graficul suprapune cursul oficial real, cursul prognozat de modelul câștigător
    și intervalul de încredere (sub forma unei zone umbrite semi-transparente).

    Args:
        dates (pd.Series): Seria calendaristică (Zilele de test).
        y_true (np.ndarray): Valorile reale.
        y_pred (np.ndarray): Valorile estimate de model.
        y_lower (np.ndarray, optional): Limita inferioară a intervalului de încredere.
        y_upper (np.ndarray, optional): Limita superioară a intervalului de încredere.
        model_name (str): Numele modelului pentru afișarea în titlu și legendă.
        currency (str): Denumirea valutei (ex: 'EUR').
        save_html_path (str, optional): Dacă este setat, va salva graficul HTML la această cale.

    Returns:
        go.Figure: Obiectul Plotly Figure gata de afișare.
    """
    fig = go.Figure()

    if y_lower is None or y_upper is None:
        margin = 0.02 * y_pred
        y_lower = y_pred - margin
        y_upper = y_pred + margin

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Limita Superioară',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 176, 246, 0.2)',
        name='Interval de Încredere',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines+markers',
        name='Curs Oficial (Real)',
        line=dict(color='black', width=2),
        marker=dict(size=6, color='black')
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='lines+markers',
        name=f'Prognoză ({model_name})',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6, color='red')
    ))

    fig.update_layout(
        title=f"Prognoza Cursului {currency} (14 Zile Test) - Model: {model_name.upper()}",
        xaxis_title="Data calendaristică",
        yaxis_title=f"Valoare Curs ({currency})",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    if save_html_path:
        fig.write_html(save_html_path)

    return fig
