"""Use-case orchestration.

Services compose the core pipeline and the adapters into workflows, and they are the
only writers of state, so the REST API, the CLI, and the library all get identical
behavior by calling the same functions (``docs/structure.md``).
"""
