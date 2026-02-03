# -*- coding: utf-8 -*-
"""Vector store for portfolio and user documents."""
import os
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL,
    PERSIST_DIR,
    CHROMA_COLLECTION_PORTFOLIO,
    CHROMA_COLLECTION_DOCS,
    RAG_TOP_K,
    REQUIRED_CSV_COLUMNS,
)


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_portfolio_docs(df: pd.DataFrame) -> list[Document]:
    """Build RAG documents from portfolio DataFrame (ticker, shares, purchase_price)."""
    docs = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        shares = row["shares"]
        purchase_price = row["purchase_price"]
        cost_basis = float(shares) * float(purchase_price)
        text = (
            f"Ticker: {ticker}, Shares: {shares}, Purchase Price: ${purchase_price:.2f}, "
            f"Cost Basis: ${cost_basis:,.2f}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "portfolio.csv", "ticker": ticker},
            )
        )
    return docs


def get_or_create_vectorstore(
    portfolio_docs: list[Document] | None = None,
    collection_name: str = CHROMA_COLLECTION_PORTFOLIO,
    persist_directory: str | None = None,
):
    """Create or load a Chroma vector store. If portfolio_docs given, use them (and optional persist)."""
    persist = persist_directory or PERSIST_DIR
    Path(persist).mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    if portfolio_docs:
        return Chroma.from_documents(
            portfolio_docs,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist,
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist,
    )


def add_user_documents(
    docs: list[Document],
    vectorstore: Chroma,
) -> None:
    """Add user-uploaded documents to an existing vector store (e.g. statements)."""
    if not docs:
        return
    vectorstore.add_documents(docs)


def retrieve_portfolio(query: str, vectorstore: Chroma, k: int = RAG_TOP_K) -> str:
    """Retrieve relevant portfolio (or user doc) chunks for a query."""
    if vectorstore is None:
        return "No portfolio or documents have been loaded."
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    return "\n".join([d.page_content for d in results]) if results else "No matching data."
