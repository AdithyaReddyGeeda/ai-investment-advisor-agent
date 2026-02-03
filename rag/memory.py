# -*- coding: utf-8 -*-
"""Conversation memory for recent context (RAG over recent turns)."""
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

from config import (
    EMBEDDING_MODEL,
    PERSIST_DIR,
    CHROMA_COLLECTION_MEMORY,
    MEMORY_TOP_K,
)


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _memory_store(session_id: str):
    """Per-session in-memory or persisted store for conversation snippets. We use a simple in-memory Chroma for the session."""
    persist = Path(PERSIST_DIR) / "memory"
    persist.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=f"{CHROMA_COLLECTION_MEMORY}_{session_id}",
        embedding_function=_get_embeddings(),
        persist_directory=str(persist),
    )


def add_turn_to_memory(session_id: str, user_text: str, assistant_text: str) -> None:
    """Store one conversation turn for later retrieval. Call after each exchange."""
    if not session_id or (not user_text.strip() and not assistant_text.strip()):
        return
    store = _memory_store(session_id)
    parts = []
    if user_text.strip():
        parts.append(f"User: {user_text.strip()}")
    if assistant_text.strip():
        parts.append(f"Assistant: {assistant_text.strip()}")
    if not parts:
        return
    doc = Document(
        page_content=" ".join(parts),
        metadata={"session": session_id},
    )
    store.add_documents([doc])


def get_recent_context(session_id: str, query: str, k: int = MEMORY_TOP_K) -> str:
    """Retrieve relevant past conversation snippets for context."""
    if not session_id:
        return ""
    try:
        store = _memory_store(session_id)
        retriever = store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs]) if docs else ""
    except Exception:
        return ""
