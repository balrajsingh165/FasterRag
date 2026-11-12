"""A sample fasterRag plugin: an in-process vector database registered as ``memory``."""

from fasterrag_memory_vectordb.adapter import MemoryVectorDB

__all__ = ["MemoryVectorDB"]
