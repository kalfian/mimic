"""Accounts, sessions and authorization (docs/PLAN-auth.md).

Kept import-free on purpose (like ``app.api``): importing a submodule never pulls in the rest.
``app.pipeline`` and ``scripts/`` must never import from here.
"""
