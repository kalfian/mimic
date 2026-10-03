"""HTTP layer (PLAN §5): routers, request dependencies and the frozen response models.

Kept import-free on purpose: ``app.core`` imports ``app.api.schemas``, so importing the routers
here would create a cycle. ``app.main`` imports the router modules directly.
"""
