from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import get_query_engine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine = get_query_engine()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    response = query_engine.query(query.question)
    return {"answer": str(response)}
