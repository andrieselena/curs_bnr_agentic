import uvicorn
from fastapi import FastAPI
from src.curs_bnr.api.endpoints import router

app = FastAPI(
    title="API Curs Valutar BNR EUR/RON",
    description="Backend FastAPI pentru interfața web",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    print("Pornire server FastAPI pe portul 7772...")
    uvicorn.run("run_backend:app", host="127.0.0.1", port=7772, reload=True)
