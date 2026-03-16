# -*- coding: utf-8 -*-
"""Conversation memory for recent context (RAG over recent turns)."""
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import time

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
    store = Chroma(
        collection_name=f"{CHROMA_COLLECTION_MEMORY}_{session_id}",
        embedding_function=_get_embeddings(),
        persist_directory=str(persist),
    )
    # Tag collection with creation time for later cleanup
    try:
        store._client.set_collection_metadata(  # type: ignore[attr-defined]
            name=f"{CHROMA_COLLECTION_MEMORY}_{session_id}",
            metadata={"created_at": time.time()},
        )
    except Exception:
        pass
    return store


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


def cleanup_old_sessions(max_age_seconds: int = 24 * 3600) -> None:
    """Delete memory collections older than max_age_seconds based on metadata."""
    persist = Path(PERSIST_DIR) / "memory"
    persist.mkdir(parents=True, exist_ok=True)
    try:
        client = Chroma(
            collection_name=f"{CHROMA_COLLECTION_MEMORY}_cleanup_probe",
            embedding_function=_get_embeddings(),
            persist_directory=str(persist),
        )._client  # type: ignore[attr-defined]
    except Exception:
        return
    try:
        collections = client.list_collections()
    except Exception:
        return
    now = time.time()
    for col in collections:
        name = getattr(col, "name", "")
        if not str(name).startswith(f"{CHROMA_COLLECTION_MEMORY}_"):
            continue
        metadata = getattr(col, "metadata", {}) or {}
        created_at = metadata.get("created_at")
        if not isinstance(created_at, (int, float)):
            continue
        if now - float(created_at) > max_age_seconds:
            try:
                client.delete_collection(name=name)
            except Exception:
                continue
