"""Embed code chunks with Voyage Code 3 and store in ChromaDB."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb

from tribal_knowledge.models import CodeChunk

BATCH_SIZE = 64
VOYAGE_MODEL = "voyage-code-3"


def _batch_items(items: list[CodeChunk], batch_size: int) -> list[list[CodeChunk]]:
    """Split a list of chunks into batches of *batch_size*."""
    batches: list[list[CodeChunk]] = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def _get_collection(output_dir: Path) -> chromadb.Collection:
    """Create or get the ChromaDB collection for code chunks."""
    chroma_path = output_dir / ".tribal-knowledge" / "chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name="code_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _build_metadata(chunk: CodeChunk) -> dict[str, str | float]:
    """Build metadata dict for a single chunk."""
    return {
        "file_path": chunk.file_path,
        "entity_type": chunk.entity_type,
        "entity_name": chunk.entity_name,
        "language": chunk.language,
        "module": chunk.module or "",
        "pagerank": chunk.pagerank,
    }


def embed_and_store(
    chunks: list[CodeChunk],
    output_dir: Path,
    embedding_provider: str = "voyage",
) -> None:
    """Embed chunks and store them in ChromaDB.

    Parameters
    ----------
    chunks:
        Code chunks with context envelopes to embed and store.
    output_dir:
        Base output directory. ChromaDB will be stored at
        ``{output_dir}/.tribal-knowledge/chromadb/``.
    embedding_provider:
        Embedding provider to use. Currently supports ``"voyage"``.
        Falls back gracefully if ``VOYAGE_API_KEY`` is not set.
    """
    if not chunks:
        return

    collection = _get_collection(output_dir)

    # Try to get embeddings from Voyage
    use_embeddings = False
    all_embeddings: list[list[float]] = []

    if embedding_provider == "voyage":
        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if api_key:
            try:
                import voyageai

                vo_client = voyageai.Client()
                batches = _batch_items(chunks, BATCH_SIZE)

                for batch in batches:
                    texts = [chunk.content for chunk in batch]
                    result = vo_client.embed(
                        texts=texts,
                        model=VOYAGE_MODEL,
                        input_type="document",
                        truncation=True,
                    )
                    all_embeddings.extend(result.embeddings)

                use_embeddings = True
            except Exception as exc:
                print(f"[warning] Voyage embedding failed: {exc}")
                print("[warning] Storing documents in ChromaDB without embeddings.")
                use_embeddings = False
                all_embeddings = []
        else:
            print(
                "[warning] VOYAGE_API_KEY not set. "
                "Storing documents in ChromaDB without embeddings."
            )

    # Store in ChromaDB (in batches to avoid large payload issues)
    batches = _batch_items(chunks, BATCH_SIZE)
    embedding_offset = 0

    for batch in batches:
        ids = [chunk.chunk_id for chunk in batch]
        documents = [chunk.content for chunk in batch]
        metadatas = [_build_metadata(chunk) for chunk in batch]

        if use_embeddings:
            batch_embeddings = all_embeddings[embedding_offset : embedding_offset + len(batch)]
            embedding_offset += len(batch)
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )
        else:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
