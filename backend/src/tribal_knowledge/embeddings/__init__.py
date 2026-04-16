"""Code chunking and vector embedding pipeline."""

from tribal_knowledge.embeddings.chunker import create_chunks
from tribal_knowledge.embeddings.store import embed_and_store

__all__ = ["create_chunks", "embed_and_store"]
