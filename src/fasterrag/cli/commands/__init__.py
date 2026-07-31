"""Command implementations, grouped by what they act on.

Each handler takes the parsed arguments and a console, and returns an exit code. None of
them raise for an expected failure — an unreachable dependency, an invalid config, a failed
preflight are all *results*, reported with the code that distinguishes them.
"""
