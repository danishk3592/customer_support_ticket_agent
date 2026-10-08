from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval component with a small stable output contract."""

    RELEVANCE_THRESHOLD = 0.35

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    @staticmethod
    def _document_id(source: str, content: str) -> str:
        digest = hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()
        return f"support-{digest}"

    async def initialize(self) -> None:
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)

        db_path = Path(self.settings.vector_db_path).expanduser()
        db_path.mkdir(parents=True, exist_ok=True)

        store = Chroma(
            collection_name=self.settings.rag_collection,
            persist_directory=str(db_path),
            embedding_function=embeddings,
        )

        ids = [
            self._document_id(doc.metadata.get("source", "unknown.md"), doc.page_content)
            for doc in documents
        ]
        existing_ids = set(store.get().get("ids", []))
        new_documents = []
        new_ids = []
        for document, document_id in zip(documents, ids, strict=True):
            if document_id not in existing_ids:
                document.metadata["source"] = Path(
                    document.metadata.get("source", "unknown.md")
                ).name
                new_documents.append(document)
                new_ids.append(document_id)

        if new_documents:
            store.add_documents(new_documents, ids=new_ids)

        # Only publish a usable store after all initialization steps succeed.
        self._store = store

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []
        if self._store is None:
            raise ComponentNotReadyError("Knowledge retriever is not ready")

        requested_limit = limit if limit is not None else self.settings.rag_top_k
        requested_limit = max(1, requested_limit)

        results = self._store.similarity_search_with_relevance_scores(
            query.strip(), k=requested_limit
        )

        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for document, score in results:
            if score < self.RELEVANCE_THRESHOLD:
                continue
            content = document.page_content.strip()
            source = Path(str(document.metadata.get("source", "unknown.md"))).name
            if not content or not source:
                continue
            key = (source, content)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"content": content, "source": source})
            if len(normalized) >= requested_limit:
                break

        return normalized
