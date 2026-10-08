from pathlib import Path

import pytest

from src.config import Settings
from src.rag import retriever as retriever_module
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vector-db"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


@pytest.mark.asyncio
async def test_search_before_initialization_fails(tmp_path: Path) -> None:
    retriever = KnowledgeRetriever(make_settings(tmp_path), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping time")


@pytest.mark.asyncio
async def test_blank_search_returns_empty(tmp_path: Path) -> None:
    retriever = KnowledgeRetriever(make_settings(tmp_path), tmp_path)
    assert await retriever.search("   ") == []


class FakeStore:
    shared_ids: set[str] = set()
    added = []

    def __init__(self, *args, **kwargs):
        self.ids = set(self.__class__.shared_ids)

    def get(self):
        return {"ids": list(self.ids)}

    def add_documents(self, documents, ids):
        self.ids.update(ids)
        self.__class__.shared_ids.update(ids)
        self.__class__.added.extend(zip(documents, ids, strict=True))

    def similarity_search_with_relevance_scores(self, query, k):
        from langchain_core.documents import Document

        return [
            (Document(page_content="Shipping takes 3–5 business days.", metadata={"source": "/unsafe/shipping.md"}), 0.90),
            (Document(page_content="Unrelated low-confidence text", metadata={"source": "other.md"}), 0.20),
        ][:k]


@pytest.mark.asyncio
async def test_initialize_is_idempotent_and_search_normalizes_results(tmp_path: Path, monkeypatch) -> None:
    FakeStore.shared_ids.clear()
    FakeStore.added.clear()
    monkeypatch.setattr(retriever_module, "Chroma", FakeStore)
    monkeypatch.setattr(retriever_module, "build_embeddings", lambda settings: object())

    retriever = KnowledgeRetriever(
        make_settings(tmp_path),
        Path(__file__).resolve().parents[1] / "knowledge_base",
    )
    await retriever.initialize()
    first_count = len(FakeStore.added)
    await retriever.initialize()

    assert first_count > 0
    assert len(FakeStore.added) == first_count

    results = await retriever.search("standard shipping")
    assert results == [{"content": "Shipping takes 3–5 business days.", "source": "shipping.md"}]
