"""The RAG pipeline: parsing, chunking, retrieval, reranking, assembly, generation.

Pure domain logic. Everything here depends on adapter *interfaces* and never on a
concrete vendor, and it holds no I/O of its own beyond reading what it is handed
(``docs/structure.md``).
"""
