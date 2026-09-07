from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import build_chain


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chain = build_chain()
    yield


app = FastAPI(title="Restaurant RAG Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return AskResponse(answer=app.state.chain.invoke(req.question))
