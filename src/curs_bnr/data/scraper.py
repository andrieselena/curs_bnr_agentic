import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

from src.curs_bnr.config import CURRENCY, START_DATE, END_DATE


def extract_raw_data(
    currency: str = CURRENCY, 
    start_date: str = START_DATE, 
    end_date: str = END_DATE
) -> List[Tuple[str, str]]:
    """
    Extrage datele brute ale cursului valutar de pe site-ul BNR simulând formularul.

    Face un HTTP POST request către site cu parametrii selectați de utilizator
    și parsează tabelul HTML returnat pentru a extrage rândurile.

    Args:
        currency (str): Valuta dorită (ex: 'EUR').
        start_date (str): Data de început în formatul cerut (ex: '22/02/2020').
        end_date (str): Data de sfârșit în formatul cerut (ex: '19/03/2026').

    Returns:
        List[Tuple[str, str]]: O listă de tupluri conținând data și valoarea cursului
                               extrăse din tabelul rezultat (ex: [('03.01.2020', '4.77'), ...]).

    Raises:
        ValueError: Dacă tabelul HTML vizat nu poate fi găsit în pagina returnată.
        requests.RequestException: Dacă cererea HTTP eșuează.
    """
    url = "https://www.cursbnr.ro/curs-valutar-bnr"
    
    payload = {
        "currency": currency,
        "dataStart": start_date,
        "dataEnd": end_date
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.post(url, data=payload, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "table-currencies"})
    
    if not table:
        raise ValueError("Tabelul cu id-ul 'table-currencies' nu a fost găsit pe pagină.")
        
    raw_data = []
    
    tbody = table.find("tbody")
    if tbody:
        rows = tbody.find_all("tr")
    else:
        rows = table.find_all("tr")[1:]
        
    if not rows:
        raise ValueError("Tabelul a fost găsit, dar nu conține niciun rând de date.")
        
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 2:
            date_str = cols[0].get_text(strip=True)
            value_str = cols[1].get_text(strip=True)
            
            if date_str and value_str:
                raw_data.append((date_str, value_str))
                
    if not raw_data:
        raise ValueError("Lista de date extrase este goală.")
            
    return raw_data
