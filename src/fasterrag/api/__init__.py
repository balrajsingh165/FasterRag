"""FastAPI surface: routers, problem responses, and the application factory.

Routers are thin — they validate a request, call one service function, and shape the
response. Business logic lives in ``services`` and the RAG pipeline in ``core``; a
router that grows a decision about chunking or retrieval is a bug (``docs/structure.md``).
"""
