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


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    docs = load_menu_docs(MENU_PATH)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
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
