"""Configuration schema and loader.

``config.yaml`` drives all behavior and is safe to commit; ``.env`` holds only secrets,
referenced from config by environment-variable *name* (``docs/adr/ADR-0003``). The
schema here is the validation contract for the whole system: it validates the entire
file at startup and fails fast with an error naming the offending key, so a running
process is never misconfigured.
"""

from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings

__all__ = ["Settings", "load_settings"]
