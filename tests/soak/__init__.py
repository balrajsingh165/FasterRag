"""The soak layer of the testing pyramid (docs/testing-strategy.md §1.8).

The harness is here and runs for a configurable duration; the 24 h run it exists for has
never been performed, because it needs the isolated reference hardware TASK-0084 waits on.
What ships is the measurement and its decision rule, proven on a short run: a long run is
the same code with a larger ``FASTERRAG_SOAK_SECONDS``.
"""
