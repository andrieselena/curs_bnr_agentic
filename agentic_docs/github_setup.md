# Ghid Publicare pe GitHub

Acest document descrie pașii necesari pentru publicarea proiectului **Curs Valutar BNR - Prognoză EUR/RON** pe GitHub.

## Pași pentru publicare

Deschide un terminal în rădăcina proiectului și execută următoarele comenzi:

```bash
git init
git status
git add .
git commit -m "Initial commit - BNR EUR/RON full stack forecasting pipeline"
git branch -M main
git remote add origin https://github.com/USER/curs-bnr-agentic.git
git push -u origin main
```

## Recomandări

- Înlocuiește `USER` cu username-ul tău real de GitHub.
- Rulează `git status` înainte de commit pentru a verifica fișierele incluse.
- Fișierele generate automat precum `__pycache__/` și `*.pyc` sunt excluse prin `.gitignore`.
- Nu include fișiere temporare, cache-uri sau date sensibile.
- Fișierele din `data/` și `outputs/` sunt păstrate intenționat pentru reproductibilitate.
