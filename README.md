# Restaurant AI Assistant

A menu Q&A assistant for **Vito’s Restaurant**. It answers questions about pizzas, salads, drinks, and prices using retrieval-augmented generation (RAG): the menu is split into one chunk per item, embedded, and searched before the language model replies.

The live app is hosted at **[https://ravindu.fun](https://ravindu.fun)**.

## What it does

- Loads `data/menu.docx` and builds **one vector per menu item** (not whole categories).
- Stores embeddings in **FAISS** (`data/faiss_index/`) so the menu is not re-embedded on every start.
- Serves **FastAPI** `POST /ask` for a React (or other) UI.
- Uses **LangGraph** so follow-ups like “is it spicy?” stay on the same conversation when the client sends `session_id`. Older turns are summarized; the last few messages stay in full.

The model is **OpenAI** (`gpt-4o-mini` for answers, `text-embedding-3-small` for vectors). Prompts live in `prompts/` (`system.txt`, `human.txt`, `summarize.txt`, `rewrite.txt`).

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | `{ "status": "ok" }` |
| `POST` | `/ask` | Question in, answer out |

Request:

```json
{
  "question": "How much is Beef BBQ pizza?",
  "session_id": "optional-uuid-from-the-ui"
}
```

Response:

```json
{
  "answer": "The Beef BBQ pizza is Rs. 1980 for medium (9 inch) and Rs. 2980 for large (12 inch).",
  "session_id": "same-or-new-uuid"
}
```

Send the same `session_id` on the next message so the thread continues. Omit it to start a new conversation.

Interactive docs when running locally: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Local setup

1. Python 3.12+ and an OpenAI key.

2. Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=sk-...
```

3. Install and run from the project root:

```text
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First start loads or builds the FAISS index. If you change `data/menu.docx`, the index rebuilds when it is older than the Word file (`python -m app.rag` also rebuilds it).

## Project layout

```text
app/           FastAPI app, menu parser, RAG + LangGraph
data/          menu.docx, FAISS index, optional diagrams
notebooks/     Step-by-step notebook (same ideas as the API)
prompts/       System, rewrite, and summarize prompts
```

Conversation checkpoints go to `data/checkpoints.sqlite` (not committed). On a free host the file can disappear after a cold start.

## Notebook

See `notebooks/restaurant_rag_assistant.ipynb` for the same pipeline broken into cells (chunks, embeddings, then LangGraph). Set `PROJECT_ROOT` if Jupyter’s working directory is not the repo root.
