"""The load layer of the testing pyramid (docs/testing-strategy.md §1.7).

Built, never run against fasterRag. There is no isolated reference hardware in this project
yet (TASK-0084), so every module here is tooling: it produces a report, and the report is
only a measurement when the operator procedure in docs/performance.md §5 was followed.
"""
