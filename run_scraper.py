import sys

from src.curs_bnr.config import CURRENCY, START_DATE, END_DATE, CSV_PATH
from src.curs_bnr.data.scraper import extract_raw_data
from src.curs_bnr.data.processing import format_and_validate_data


def main() -> None:
    """
    Orchestrează fluxul de extragere, procesare și salvare a cursului valutar.
    """
    print(f"Începe procesul pentru {CURRENCY} din {START_DATE} până în {END_DATE}...")

    try:
        raw_data = extract_raw_data(CURRENCY, START_DATE, END_DATE)
        print(f"Extragere completă: {len(raw_data)} rânduri brute obținute.")

        df_processed = format_and_validate_data(raw_data, CURRENCY)
        print(f"Procesare completă: {len(df_processed)} rânduri valide sortate.")

        df_processed.to_csv(CSV_PATH, index=False)
        print(f"Datele au fost salvate cu succes în fișierul: {CSV_PATH}")

    except ValueError as ve:
        print(f"Eroare de procesare sau validare: {ve}", file=sys.stderr)
    except Exception as e:
        print(f"Eroare de sistem sau rețea: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
