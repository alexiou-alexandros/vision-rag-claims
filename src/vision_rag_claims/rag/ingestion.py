"""
Ingest policy markdown files into Chroma.

Usage:
    uv run python -m vision_rag_claims.rag.ingestion          # skip if already done
    uv run python -m vision_rag_claims.rag.ingestion --force  # re-ingest from scratch
"""

import argparse
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from vision_rag_claims.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "insurance_policies"
_MARKER = Path(str(settings.chroma_persist_dir)) / ".ingested"


def _load_documents() -> list[dict]:
    docs = []
    for md_file in sorted(Path(str(settings.policies_dir)).glob("*.md")):
        docs.append({"path": md_file, "name": md_file.stem, "text": md_file.read_text(encoding="utf-8")})
    return docs


def _split_document(text: str, source_name: str) -> list:
    """Two-stage split: preserve markdown section context, then cap chunk size."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for hdoc in header_splitter.split_text(text):
        for chunk in char_splitter.split_documents([hdoc]):
            chunk.metadata["source_document"] = source_name
            chunks.append(chunk)
    return chunks


def ingest(force: bool = False) -> None:
    if _MARKER.exists() and not force:
        logger.info("Policies already ingested. Use --force to re-ingest.")
        return

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    persist_dir = str(settings.chroma_persist_dir)

    if force and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
        logger.info("Removed existing Chroma store.")

    docs = _load_documents()
    if not docs:
        raise FileNotFoundError(f"No .md files found in {settings.policies_dir}")

    all_chunks = []
    for doc in docs:
        chunks = _split_document(doc["text"], doc["name"])
        all_chunks.extend(chunks)
        logger.info("  %s: %d chunks", doc["name"], len(chunks))

    logger.info("Total chunks: %d — embedding and storing...", len(all_chunks))

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.touch()
    logger.info("Done. Chroma persisted to '%s'.", persist_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest policy documents into Chroma.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already done.")
    args = parser.parse_args()
    ingest(force=args.force)
