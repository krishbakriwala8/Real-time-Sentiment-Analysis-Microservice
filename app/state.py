"""
app/state.py
────────────
Shared mutable application state.

This module exists to avoid circular imports between app.main (which sets
startup_time) and app.api.routes (which reads it for the /health endpoint).
"""

from __future__ import annotations

# Set by the lifespan handler in app.main once the service is ready.
startup_time: float = 0.0
