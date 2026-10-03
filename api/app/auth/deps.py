"""Request dependencies for authentication / authorization (PLAN-auth §3.3, §4, §7).

Dependency chain (check order on every protected route, after the Origin middleware)::

    get_session_user  -> 401 unauthenticated            (session cookie -> active user)
    get_current_user  -> 403 password_change_required   (full session)
    require_admin     -> 403 forbidden                  (role admin)

``get_optional_user`` is the never-raising variant for public routes (``/api/health``): the
user of a *full* session, else None.

Tests replace ``get_session_user`` and ``get_optional_user`` via ``app.dependency_overrides``
(``tests/auth_helpers.py::as_user``); ``get_current_user`` / ``require_admin`` then run their
real checks on the injected user.

Session resolution is ``auth.sessions.resolve(<cookie>)`` (``app/auth/sessions.py``): one
``sessions JOIN users`` read; role and flags come from the users row at request time. The deps
are plain ``def`` so the SQLite read runs on the threadpool.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.auth.models import AuthServices, UserRecord
from app.core.errors import ErrorCode, PipelineError

#: Session cookie name (A2). A constant, not a setting.
SESSION_COOKIE = "mimic_session"


def get_auth(request: Request) -> AuthServices:
    """``app.state.auth`` (built by ``app.main``'s lifespan)."""
    auth: AuthServices | None = getattr(request.app.state, "auth", None)
    if auth is None:  # only possible if the app was used without its lifespan
        raise PipelineError(ErrorCode.INTERNAL_ERROR, "Server is not initialized.")
    return auth


AuthDep = Annotated[AuthServices, Depends(get_auth)]


def session_token(request: Request) -> str | None:
    """The raw session cookie value, if any. Never log it."""
    token = request.cookies.get(SESSION_COOKIE)
    return token or None


def _resolve_user(request: Request, auth: AuthServices) -> UserRecord | None:
    token = session_token(request)
    if token is None:
        return None
    resolved = auth.sessions.resolve(token)
    return resolved[1] if resolved is not None else None


def get_session_user(request: Request, auth: AuthDep) -> UserRecord:
    """User of a valid session (forced password change allowed). 401 ``unauthenticated``
    otherwise (no cookie, unknown/expired session, disabled user)."""
    user = _resolve_user(request, auth)
    if user is None:
        raise PipelineError(ErrorCode.UNAUTHENTICATED)
    return user


def get_optional_user(request: Request, auth: AuthDep) -> UserRecord | None:
    """User of a valid *full* session, else None. Never raises for auth reasons (a forced
    password change counts as no full session)."""
    user = _resolve_user(request, auth)
    return user if user is not None and user.has_full_access else None


SessionUserDep = Annotated[UserRecord, Depends(get_session_user)]
OptionalUserDep = Annotated[UserRecord | None, Depends(get_optional_user)]


def get_current_user(user: SessionUserDep) -> UserRecord:
    """Full session: 403 ``password_change_required`` while a forced change is pending."""
    if user.must_change_password:
        raise PipelineError(ErrorCode.PASSWORD_CHANGE_REQUIRED)
    return user


CurrentUserDep = Annotated[UserRecord, Depends(get_current_user)]


def require_admin(user: CurrentUserDep) -> UserRecord:
    """Admin with a full session: 403 ``forbidden`` for users."""
    if not user.is_admin:
        raise PipelineError(ErrorCode.FORBIDDEN)
    return user


AdminDep = Annotated[UserRecord, Depends(require_admin)]
