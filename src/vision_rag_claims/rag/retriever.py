from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from vision_rag_claims.config import settings
from vision_rag_claims.rag.ingestion import COLLECTION_NAME
from vision_rag_claims.schemas import PolicyChunk


class PolicyRetriever:
    """Thin wrapper around Chroma that returns typed PolicyChunk objects."""

    def __init__(self) -> None:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> list[PolicyChunk]:
        results = self._store.similarity_search(query, k=k or settings.retrieval_k)
        chunks = []
        for doc in results:
            meta = doc.metadata
            chunks.append(PolicyChunk(
                content=doc.page_content,
                source_document=meta.get("source_document", "unknown"),
                section=meta.get("h2") or meta.get("h1") or meta.get("h3") or "",
                metadata=meta,
            ))
        return chunks
