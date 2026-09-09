import os
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.menu import load_menu_docs

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "data" / "menu.docx"
PROMPT_DIR = ROOT / "prompts"
INDEX_DIR = ROOT / "data" / "faiss_index"
CHECKPOINT_PATH = ROOT / "data" / "checkpoints.sqlite"

MESSAGE_THRESHOLD = 6
KEEP_LAST = 4


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _index_is_fresh() -> bool:
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists() or not MENU_PATH.exists():
        return False
    return index_file.stat().st_mtime >= MENU_PATH.stat().st_mtime


def load_or_create_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if _index_is_fresh():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs = load_menu_docs(MENU_PATH)
    vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    search_query: str
    context: str


def build_graph():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    system_text = load_prompt("system.txt")
    summarize_text = load_prompt("summarize.txt")
    rewrite_text = load_prompt("rewrite.txt")
    retriever = load_or_create_vectorstore().as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

    def maybe_summarize(state: GraphState) -> dict:
        messages = state["messages"]
        if len(messages) <= MESSAGE_THRESHOLD:
            return {}

        old = messages[:-KEEP_LAST]
        existing = state.get("summary") or ""
        transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
        result = llm.invoke(
            [
                SystemMessage(content=summarize_text),
                HumanMessage(
                    content=f"Previous summary:\n{existing}\n\nOlder messages:\n{transcript}"
                ),
            ]
        )
        return {
            "summary": result.content,
            "messages": [RemoveMessage(id=m.id) for m in old],
        }

    def rewrite_query(state: GraphState) -> dict:
        messages = state["messages"]
        last = messages[-1].content
        if len(messages) == 1 and not state.get("summary"):
            return {"search_query": last}

        recent = "\n".join(f"{m.type}: {m.content}" for m in messages)
        payload = (
            f"Latest user message:\n{last}\n\n"
            f"Recent messages (most recent last). Resolve it/that from these first:\n{recent}\n\n"
            f"Older summary (background only; do not use for it/that unless the latest message clearly refers back):\n"
            f"{state.get('summary') or '(none)'}"
        )
        result = llm.invoke(
            [
                SystemMessage(content=rewrite_text),
                HumanMessage(content=payload),
            ]
        )
        return {"search_query": result.content.strip()}

    def retrieve_menu(state: GraphState) -> dict:
        docs = retriever.invoke(state["search_query"])
        return {"context": format_docs(docs)}

    def generate_answer(state: GraphState) -> dict:
        system = system_text
        if state.get("summary"):
            system += f"\n\nEarlier conversation summary:\n{state['summary']}"
        system += f"\n\nMenu context:\n{state['context']}"
        result = llm.invoke([SystemMessage(content=system), *state["messages"]])
        return {"messages": [AIMessage(content=result.content)]}

    builder = StateGraph(GraphState)
    builder.add_node("summarize", maybe_summarize)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("retrieve", retrieve_menu)
    builder.add_node("generate", generate_answer)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


def ask(graph, question: str, session_id: str | None = None) -> tuple[str, str]:
    thread_id = session_id or str(uuid.uuid4())
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content, thread_id


if __name__ == "__main__":
    load_or_create_vectorstore()
    print(f"Saved vector index to {INDEX_DIR}")
