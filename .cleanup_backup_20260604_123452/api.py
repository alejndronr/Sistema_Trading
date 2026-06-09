import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("+aiosqlite", "")
engine = create_engine(db_url)
Base = declarative_base()

class ManualTrade(Base):
    __tablename__ = 'manual_trades_v2'
    id = Column(Integer, primary_key=True)
    coinId = Column(String)
    type = Column(String, default="COMPRA")
    amount = Column(Float)
    buyPrice = Column(Float)
    date = Column(String)
    exchange = Column(String)
    notes = Column(String)

Session = sessionmaker(bind=engine)
app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

class TradeIn(BaseModel):
    coinId: str
    type: str
    amount: float
    buyPrice: float
    date: str
    exchange: str
    notes: str

@app.get("/")
def serve_dashboard():
    with open("web/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/investments")
def get_investments():
    session = Session()
    trades = session.query(ManualTrade).all()
    res = [{"id": t.id, "coinId": t.coinId, "type": t.type, "amount": t.amount, "buyPrice": t.buyPrice, "date": t.date, "exchange": t.exchange, "notes": t.notes} for t in trades]
    session.close()
    return res

@app.post("/api/investments")
def add_investment(trade: TradeIn):
    session = Session()
    new_trade = ManualTrade(**trade.dict())
    session.add(new_trade)
    session.commit()
    session.close()
    return {"status": "ok"}

@app.delete("/api/investments/{trade_id}")
def delete_investment(trade_id: int):
    session = Session()
    trade = session.query(ManualTrade).filter(ManualTrade.id == trade_id).first()
    if trade:
        session.delete(trade)
        session.commit()
    session.close()
    return {"status": "ok"}

@app.get("/api/bot-trades")
def get_bot_trades():
    try:
        df = pd.read_sql("SELECT * FROM trades", engine)
        df = df.fillna(0)
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient="records")
    except:
        return []
