"""``/api/auth/*`` routes (PLAN-auth §3.3, A1–A6, A11).

Every response here carries ``Cache-Control: no-store`` (``NoStoreMiddleware`` in
``app.main``). Routes are plain ``def`` so scrypt and SQLite run on FastAPI's threadpool, not
the event loop.

Logging: ids only. Never a password, token, token hash or username of a failed attempt.
"""

from __future__ import annotations

import itertools
import logging

from fastapi import APIRouter, Request, Response

from app.api.schemas import AuthStatus, ChangePasswordRequest, ErrorBody, LoginRequest, Me
from app.auth import passwords
from app.auth.deps import AuthDep, SessionUserDep, session_token
from app.auth.models import AuthServices, UserRecord
from app.auth.policy import check_password_policy
from app.auth.sessions import clear_session_cookie, set_session_cookie
from app.core.errors import ErrorCode, PipelineError

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_failed_logins = itertools.count(1)


def _errors(*codes: int) -> dict[int | str, dict[str, object]]:
    return {code: {"model": ErrorBody} for code in codes}


def to_me(user: UserRecord) -> Me:
    return Me(
        id=user.id,
        username=user.username,
        role=user.role.value,
        must_change_password=user.must_change_password,
        created_at=user.created_at,
    )


def client_ip(request: Request) -> str:
    """Peer address. ``X-Forwarded-For`` is deliberately not trusted (A6)."""
    return request.client.host if request.client else "unknown"


def _throttle(auth: AuthServices, ip: str, key: str) -> None:
    wait = auth.limiter.retry_after(ip, key)
    if wait is not None:
        raise PipelineError(ErrorCode.TOO_MANY_ATTEMPTS, headers={"Retry-After": str(wait)})


def _start_session(response: Response, auth: AuthServices, user_id: str) -> None:
    token = auth.sessions.create(user_id)
    set_session_cookie(response, token, auth.settings)


@router.get("/api/auth/status", response_model=AuthStatus)
def auth_status(auth: AuthDep) -> AuthStatus:
    """Public. ``setup_required`` while no active admin exists (A11)."""
    return AuthStatus(setup_required=not auth.users.has_active_admin())


@router.post(
    "/api/auth/login",
    response_model=Me,
    responses=_errors(401, 403, 409, 422, 429),
)
def login(body: LoginRequest, request: Request, response: Response, auth: AuthDep) -> Me:
    if not auth.users.has_active_admin():
        raise PipelineError(ErrorCode.SETUP_REQUIRED)
    ip, username = client_ip(request), body.username
    _throttle(auth, ip, username)

    user = auth.users.get_by_username(username)
    if user is None:
        passwords.dummy_verify(body.password)  # same cost as a real check (enumeration)
        ok = False
    else:
        ok = passwords.verify_password(body.password, user.password_hash)
    if user is None or not ok:
        auth.limiter.record_failure(ip, username)
        log.info("auth: failed sign-in (%d since start)", next(_failed_logins))
        raise PipelineError(ErrorCode.INVALID_CREDENTIALS)
    if not user.is_active:  # only revealed after a correct password
        raise PipelineError(ErrorCode.ACCOUNT_DISABLED)

    auth.limiter.reset(ip, username)
    if passwords.needs_rehash(user.password_hash):
        auth.users.set_password(
            user.id,
            passwords.hash_password(body.password),
            must_change=user.must_change_password,
        )
    old = session_token(request)
    if old is not None:
        auth.sessions.revoke(old)
    auth.sessions.purge_expired()
    _start_session(response, auth, user.id)
    auth.users.record_login(user.id)
    log.info("auth: user %s signed in", user.id)
    return to_me(user)


@router.post("/api/auth/logout", status_code=204, response_class=Response)
def logout(request: Request, auth: AuthDep) -> Response:
    """Idempotent: 204 with an expiring cookie whether or not a session existed."""
    token = session_token(request)
    if token is not None:
        resolved = auth.sessions.resolve(token)
        auth.sessions.revoke(token)
        if resolved is not None:
            log.info("auth: user %s signed out", resolved[1].id)
    response = Response(status_code=204)
    clear_session_cookie(response, auth.settings)
    return response


@router.get("/api/auth/me", response_model=Me, responses=_errors(401))
def me(user: SessionUserDep) -> Me:
    """Current account (also while a password change is pending)."""
    return to_me(user)


@router.post(
    "/api/auth/password",
    response_model=Me,
    responses=_errors(401, 422, 429),
)
def change_password(
    body: ChangePasswordRequest,
    request: Request,
    response: Response,
    user: SessionUserDep,
    auth: AuthDep,
) -> Me:
    """Change the own password. Revokes every session of the account, then starts a new one."""
    ip, key = client_ip(request), f"user:{user.id}"
    _throttle(auth, ip, key)
    if not passwords.verify_password(body.current_password, user.password_hash):
        auth.limiter.record_failure(ip, key)
        raise PipelineError(ErrorCode.CURRENT_PASSWORD_INCORRECT)
    violation = check_password_policy(
        body.new_password, user.username, current_password=body.current_password
    )
    if violation is not None:
        raise PipelineError(ErrorCode.WEAK_PASSWORD, violation)

    auth.limiter.reset(ip, key)
    auth.users.set_password(user.id, passwords.hash_password(body.new_password), must_change=False)
    auth.sessions.revoke_all(user.id)
    _start_session(response, auth, user.id)
    log.info("auth: user %s changed their password", user.id)
    updated = auth.users.get(user.id)
    if updated is None:  # deleted concurrently
        raise PipelineError(ErrorCode.UNAUTHENTICATED)
    return to_me(updated)
