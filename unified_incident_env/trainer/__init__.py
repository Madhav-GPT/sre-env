"""Trainer package namespace.

This package intentionally avoids eager importing of legacy trainer flows so the
honest v2 environment can reuse shell utilities without pulling in deprecated
benchmark-specific modules at import time.
"""

__all__: list[str] = []
