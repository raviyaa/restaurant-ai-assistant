import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.menu import load_menu_docs

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "data" / "menu.docx"
PROMPT_DIR = ROOT / "prompts"
INDEX_DIR = ROOT / "data" / "faiss_index"


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


def build_chain():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    vectorstore = load_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")),
            ("human", (PROMPT_DIR / "human.txt").read_text(encoding="utf-8")),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=120)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    load_or_create_vectorstore()
    print(f"Saved vector index to {INDEX_DIR}")
