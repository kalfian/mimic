"""``/api/admin/*`` routes: user management (PLAN-auth §3.3, U2, A5, A13, A14).

Every route needs ``AdminDep`` (401 anonymous, 403 ``password_change_required`` while forced,
403 ``forbidden`` for users). Every response carries ``Cache-Control: no-store``
(``NoStoreMiddleware`` in ``app.main``). Unknown or malformed user ids → 404 ``not_found``.

Guards: ``self_action_forbidden`` (A13: disable / reset / delete the own account) is checked
here; ``last_admin`` is atomic inside :class:`app.auth.users.SqliteUserStore`. Role change,
disable and reset revoke all of the target's sessions.

Temporary passwords (A5) are server-generated, returned once and never logged. Audit lines
contain ids only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from app.api.deps import ServicesDep
from app.api.schemas import (
    AdminUser,
    AdminUserList,
    CreateUserRequest,
    ErrorBody,
    TemporaryPassword,
    UpdateUserRequest,
)
from app.auth import passwords
from app.auth.deps import AdminDep, AuthDep
from app.auth.models import AuthServices, Role, UserRecord
from app.auth.policy import is_valid_user_id
from app.core.errors import ErrorCode, PipelineError

log = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])

_USER_NOT_FOUND = "This user does not exist."


def _errors(*codes: int) -> dict[int | str, dict[str, object]]:
    return {code: {"model": ErrorBody} for code in (401, 403, *codes)}


def to_admin_user(user: UserRecord, job_count: int) -> AdminUser:
    return AdminUser(
        id=user.id,
        username=user.username,
        role=user.role.value,
        is_active=user.is_active,
        must_change_password=user.must_change_password,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        job_count=job_count,
    )


def _target(auth: AuthServices, user_id: str) -> UserRecord:
    user = auth.users.get(user_id) if is_valid_user_id(user_id) else None
    if user is None:
        raise PipelineError(ErrorCode.NOT_FOUND, _USER_NOT_FOUND)
    return user


def _with_job_count(auth: AuthServices, user: UserRecord) -> AdminUser:
    # The admin list is small (no pagination, §3.1); reuse its count query.
    count = next((n for u, n in auth.users.list_with_job_counts() if u.id == user.id), 0)
    return to_admin_user(user, count)


def _not_self(admin: UserRecord, target: UserRecord) -> None:
    if admin.id == target.id:
        raise PipelineError(ErrorCode.SELF_ACTION_FORBIDDEN)


@router.get("/api/admin/users", response_model=AdminUserList, responses=_errors())
def list_users(admin: AdminDep, auth: AuthDep) -> AdminUserList:
    return AdminUserList(items=[to_admin_user(u, n) for u, n in auth.users.list_with_job_counts()])


@router.post(
    "/api/admin/users",
    status_code=201,
    response_model=TemporaryPassword,
    responses=_errors(409, 422),
)
def create_user(body: CreateUserRequest, admin: AdminDep, auth: AuthDep) -> TemporaryPassword:
    temporary = passwords.generate_temporary_password()
    user = auth.users.create(
        body.username,
        passwords.hash_password(temporary),
        Role(body.role),
        must_change_password=True,
        created_by=admin.id,
    )
    log.info("admin %s created user %s (%s)", admin.id, user.id, user.role.value)
    return TemporaryPassword(user=to_admin_user(user, 0), temporary_password=temporary)


@router.patch(
    "/api/admin/users/{user_id}",
    response_model=AdminUser,
    responses=_errors(404, 409, 422),
)
def update_user(user_id: str, body: UpdateUserRequest, admin: AdminDep, auth: AuthDep) -> AdminUser:
    target = _target(auth, user_id)
    if body.is_active is False:
        _not_self(admin, target)
    updated = auth.users.update(
        target.id,
        role=Role(body.role) if body.role is not None else None,
        is_active=body.is_active,
    )
    role_changed = updated.role is not target.role
    disabled = target.is_active and not updated.is_active
    if role_changed or disabled:
        auth.sessions.revoke_all(target.id)
    if role_changed:
        log.info("admin %s changed role of user %s to %s", admin.id, target.id, updated.role.value)
    if updated.is_active != target.is_active:
        action = "enabled" if updated.is_active else "disabled"
        log.info("admin %s %s user %s", admin.id, action, target.id)
    return _with_job_count(auth, updated)


@router.post(
    "/api/admin/users/{user_id}/reset-password",
    response_model=TemporaryPassword,
    responses=_errors(404, 409),
)
def reset_password(user_id: str, admin: AdminDep, auth: AuthDep) -> TemporaryPassword:
    target = _target(auth, user_id)
    _not_self(admin, target)
    temporary = passwords.generate_temporary_password()
    auth.users.set_password(target.id, passwords.hash_password(temporary), must_change=True)
    auth.sessions.revoke_all(target.id)
    log.info("admin %s reset the password of user %s", admin.id, target.id)
    updated = auth.users.get(target.id)
    if updated is None:  # deleted concurrently
        raise PipelineError(ErrorCode.NOT_FOUND, _USER_NOT_FOUND)
    return TemporaryPassword(user=_with_job_count(auth, updated), temporary_password=temporary)


@router.delete(
    "/api/admin/users/{user_id}",
    status_code=204,
    response_class=Response,
    responses=_errors(404, 409),
)
def delete_user(user_id: str, admin: AdminDep, auth: AuthDep, services: ServicesDep) -> Response:
    """Delete sessions, job rows and the user in one transaction, then the job directories
    (best effort). A job still running is cleaned up by the runner's deleted-job guard (A14)."""
    target = _target(auth, user_id)
    _not_self(admin, target)
    job_ids = auth.users.delete(target.id)
    for job_id in job_ids:
        try:
            services.storage.delete_job(job_id)
        except Exception as exc:  # best effort: the row is gone either way
            log.warning("admin: could not delete files of job %s (%s)", job_id, type(exc).__name__)
    log.info("admin %s deleted user %s (%d jobs)", admin.id, target.id, len(job_ids))
    return Response(status_code=204)
