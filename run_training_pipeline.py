import sys

from prophet import Prophet

from src.curs_bnr.config import CSV_PATH, VALUE_COL, CURRENCY, MODELS_DIR, REPORTS_DIR
from src.curs_bnr.features.engineering import prepare_ml_data
from src.curs_bnr.optimization.tuning import (
    optimize_sarima_gridsearch,
    optimize_prophet_gridsearch,
    optimize_xgboost_optuna,
)
from src.curs_bnr.training.trainers import (
    train_sarima,
    train_xgboost,
)
from src.curs_bnr.evaluation.metrics import (
    compare_and_save_best_model,
    evaluate_prophet,
    evaluate_xgboost,
    evaluate_sarima,
)
from src.curs_bnr.visualization.plots import create_forecast_plot


def main() -> None:
    """
    Orchestrează fluxul complet de Machine Learning.
    """
    print("1. Se încarcă și se preprocesează datele...")
    try:
        data_dict = prepare_ml_data(str(CSV_PATH), VALUE_COL)
    except Exception as e:
        print(f"Eroare la preprocesare: {e}", file=sys.stderr)
        return

    train_base = data_dict["train_base"]
    train_xgb = data_dict["train_xgb"]

    print("\n2. [TUNING] Căutarea hiperparametrilor optimi...")
    print(" -> Rulare GridSearch pentru SARIMA...")
    best_order, best_seasonal, score_sarima = optimize_sarima_gridsearch(train_base, VALUE_COL, n_splits=3)
    print(f"    => Parametri optimi SARIMA: order={best_order}, seasonal={best_seasonal} (RMSE CV: {score_sarima:.4f})")

    print(" -> Rulare GridSearch pentru Prophet...")
    best_params_prophet, score_prophet = optimize_prophet_gridsearch(train_base, VALUE_COL, n_splits=3)
    print(f"    => Parametri optimi Prophet: {best_params_prophet} (RMSE CV: {score_prophet:.4f})")

    print(" -> Rulare Optuna pentru XGBoost...")
    best_params_xgb = optimize_xgboost_optuna(train_xgb, VALUE_COL, n_trials=15, n_splits=3)
    print(f"    => Parametri optimi XGBoost: {best_params_xgb}")

    print("\n3. Se reantrenează modelele folosind hiperparametrii optimi...")
    try:
        model_sarima = train_sarima(train_base, VALUE_COL, order=best_order, seasonal_order=best_seasonal)
        model_xgb = train_xgboost(train_xgb, VALUE_COL, xgb_params=best_params_xgb)

        df_prophet = train_base[["Data", VALUE_COL]].rename(columns={"Data": "ds", VALUE_COL: "y"})
        model_prophet = Prophet(daily_seasonality=False, yearly_seasonality=True, **best_params_prophet)
        model_prophet.fit(df_prophet)

        models_dict = {
            "prophet": model_prophet,
            "xgboost": model_xgb,
            "sarima": model_sarima
        }
    except Exception as e:
        print(f"Eroare la reantrenare: {e}", file=sys.stderr)
        return

    print("\n4. Se evaluează modelele pe setul de test și se alege câștigătorul...")
    save_model_path = str(MODELS_DIR / "model_castigator.pkl")
    eval_report = compare_and_save_best_model(
        models_dict,
        data_dict,
        VALUE_COL,
        save_path=save_model_path
    )

    best_name = eval_report["best_model_name"]
    best_model = models_dict[best_name]

    print("\n--- Rezultate Evaluare (Set de Test: ultimele 14 zile) ---")
    for m_name, metrics in eval_report["metrics"].items():
        print(f"Model: {m_name.upper()}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("-" * 48)
    print(f"Modelul câștigător absolut este: {best_name.upper()}")
    print(f"Fișierul modelului antrenat salvat la: {eval_report['save_path']}\n")

    print("5. Se generează graficul interactiv pentru modelul câștigător...")
    if best_name == "prophet":
        test_df = data_dict["test_base"]
        y_pred = evaluate_prophet(best_model, test_df)
    elif best_name == "xgboost":
        test_df = data_dict["test_xgb"]
        y_pred = evaluate_xgboost(best_model, test_df, VALUE_COL)
    elif best_name == "sarima":
        test_df = data_dict["test_base"]
        y_pred = evaluate_sarima(best_model, test_df)
    else:
        print(f"Nu se poate genera graficul. Model necunoscut: {best_name}", file=sys.stderr)
        return

    dates = test_df["Data"]
    y_true = test_df[VALUE_COL].values
    save_html_path = str(REPORTS_DIR / "grafic_prognoza_final.html")

    try:
        create_forecast_plot(
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            model_name=best_name.upper(),
            currency=CURRENCY,
            save_html_path=save_html_path
        )
        print(f"Graficul a fost generat și salvat cu succes: {save_html_path}")
    except Exception as e:
        print(f"Eroare la generarea graficului: {e}", file=sys.stderr)

    print("\n6. Se salvează rezultatele antrenării în baza de date SQLite...")
    try:
        from src.curs_bnr.database.crud import save_training_results_to_db
        import json
        
        params_str = "{}"
        if best_name == "xgboost":
            params_str = json.dumps(best_params_xgb)
        elif best_name == "prophet":
            params_str = json.dumps(best_params_prophet)
        elif best_name == "sarima":
            params_str = json.dumps({"order": best_order, "seasonal_order": best_seasonal})
        else:
            params_str = json.dumps({"note": "Parametri indisponibili pentru acest model"})
            
        best_metrics = eval_report["metrics"][best_name]
        
        save_training_results_to_db(
            model_name=best_name.upper(),
            version="1.0",
            hyperparameters=params_str,
            metrics=best_metrics,
            forecast_dates=dates.astype(str).tolist(),
            forecast_values=y_pred.tolist()
        )
        print("Datele au fost salvate cu succes în tabelele SQLite (training_runs, metrics, forecasts).")
    except Exception as e:
        print(f"Atenție: Nu s-au putut salva rezultatele în baza de date: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
