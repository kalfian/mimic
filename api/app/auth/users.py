"""User accounts in SQLite (PLAN-auth §2, U2, A10, A17).

Every read-check-write sequence runs in one ``BEGIN IMMEDIATE`` transaction
(:func:`app.core.db.transaction`), so the **last-active-admin guard** is atomic: two admins
demoting / disabling / deleting each other concurrently can't both succeed.

Job rows are touched only through the connection-level helpers in :mod:`app.core.jobstore`
(``delete_jobs_of_owner``, ``claim_orphan_jobs``) on this store's transaction. Directories are
the caller's job (after commit).

Guards raise ``PipelineError``: ``not_found``, ``last_admin``, ``username_taken``,
``invalid_request`` (username not matching A17).
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.auth import policy
from app.auth.models import USER_COLUMNS, Role, UserRecord, user_from_row
from app.core import db
from app.core.errors import ErrorCode, PipelineError
from app.core.jobstore import claim_orphan_jobs, delete_jobs_of_owner

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def _is_active_admin(role: Role, is_active: bool) -> bool:
    return role is Role.ADMIN and is_active


class SqliteUserStore:
    """:class:`app.auth.models.UserStore` on ``data/mimic.db`` (table ``users``)."""

    def __init__(
        self, db_path: Path, *, clock: Clock = utcnow, timeout_s: float = db.DEFAULT_TIMEOUT_S
    ) -> None:
        self.db_path = Path(db_path)
        self._clock = clock
        self._timeout_s = timeout_s

    def _now(self) -> str:
        return _iso(self._clock())

    # ---- helpers on an open connection ----------------------------------------------------

    @staticmethod
    def _get(conn: sqlite3.Connection, user_id: str) -> UserRecord | None:
        row = conn.execute(f"SELECT {USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
        return user_from_row(row) if row is not None else None

    @staticmethod
    def _active_admins(conn: sqlite3.Connection, *, excluding: str | None = None) -> int:
        return conn.execute(
            "SELECT count(*) FROM users WHERE role = 'admin' AND is_active = 1 AND id IS NOT ?",
            (excluding,),
        ).fetchone()[0]

    def _insert(
        self,
        conn: sqlite3.Connection,
        username: str,
        password_hash: str,
        role: Role,
        *,
        must_change_password: bool,
        created_by: str | None,
    ) -> UserRecord:
        username = policy.normalize_username(username)
        if not policy.is_valid_username(username):
            raise PipelineError(
                ErrorCode.INVALID_REQUEST,
                "Usernames are 3-32 characters: lowercase letters, digits, '.', '_' or '-'.",
            )
        user_id = uuid.uuid4().hex
        now = self._now()
        try:
            conn.execute(
                f"INSERT INTO users ({USER_COLUMNS}) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, NULL, ?)",
                (
                    user_id,
                    username,
                    password_hash,
                    Role(role).value,
                    int(must_change_password),
                    now,
                    now,
                    now,
                    created_by,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "users.username" in str(exc):
                raise PipelineError(ErrorCode.USERNAME_TAKEN) from None
            raise
        user = self._get(conn, user_id)
        assert user is not None
        return user

    # ---- UserStore ------------------------------------------------------------------------

    def create(
        self,
        username: str,
        password_hash: str,
        role: Role,
        *,
        must_change_password: bool,
        created_by: str | None = None,
    ) -> UserRecord:
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            return self._insert(
                conn,
                username,
                password_hash,
                role,
                must_change_password=must_change_password,
                created_by=created_by,
            )

    def create_admin(self, username: str, password_hash: str) -> tuple[UserRecord, int | None]:
        """CLI bootstrap (A10): create an active admin (no forced change). If no active admin
        existed before, the new admin claims every legacy (ownerless) job in the same
        transaction. Returns ``(admin, claimed job count)``; the count is None when another
        active admin already existed (break-glass, nothing claimed)."""
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            first = self._active_admins(conn) == 0
            admin = self._insert(
                conn,
                username,
                password_hash,
                Role.ADMIN,
                must_change_password=False,
                created_by=None,
            )
            claimed = claim_orphan_jobs(conn, admin.id) if first else None
        return admin, claimed

    def get(self, user_id: str) -> UserRecord | None:
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            return self._get(conn, user_id)

    def get_by_username(self, username: str) -> UserRecord | None:
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            row = conn.execute(
                f"SELECT {USER_COLUMNS} FROM users WHERE username = ?",
                (policy.normalize_username(username),),
            ).fetchone()
        return user_from_row(row) if row is not None else None

    def list_with_job_counts(self) -> list[tuple[UserRecord, int]]:
        cols = ", ".join(f"u.{c.strip()}" for c in USER_COLUMNS.split(","))
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            rows = conn.execute(
                f"SELECT {cols}, "
                "(SELECT count(*) FROM jobs j WHERE j.owner_id = u.id) AS job_count "
                "FROM users u ORDER BY u.username"
            ).fetchall()
        return [(user_from_row(r), int(r["job_count"])) for r in rows]

    def update(
        self, user_id: str, *, role: Role | None = None, is_active: bool | None = None
    ) -> UserRecord:
        """Change role and/or active flag. No-op values return the record unchanged.

        Session revocation is the caller's job (``SessionStore.revoke_all``); a disabled user
        can't resolve a session anyway and the role is always read fresh.
        """
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            user = self._get(conn, user_id)
            if user is None:
                raise PipelineError(ErrorCode.NOT_FOUND, "This user does not exist.")
            new_role = Role(role) if role is not None else user.role
            new_active = user.is_active if is_active is None else bool(is_active)
            if (new_role, new_active) == (user.role, user.is_active):
                return user
            if (
                _is_active_admin(user.role, user.is_active)
                and not _is_active_admin(new_role, new_active)
                and self._active_admins(conn, excluding=user_id) == 0
            ):
                raise PipelineError(ErrorCode.LAST_ADMIN)
            conn.execute(
                "UPDATE users SET role = ?, is_active = ?, updated_at = ? WHERE id = ?",
                (new_role.value, int(new_active), self._now(), user_id),
            )
            updated = self._get(conn, user_id)
        assert updated is not None
        return updated

    def set_password(self, user_id: str, password_hash: str, *, must_change: bool) -> None:
        """Store a new hash. Session revocation is the caller's job."""
        now = self._now()
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ?, must_change_password = ?, "
                "password_changed_at = ?, updated_at = ? WHERE id = ?",
                (password_hash, int(must_change), now, now, user_id),
            )
        if cur.rowcount == 0:
            raise PipelineError(ErrorCode.NOT_FOUND, "This user does not exist.")

    def record_login(self, user_id: str) -> None:
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (self._now(), user_id))

    def delete(self, user_id: str) -> list[str]:
        with db.transaction(self.db_path, timeout_s=self._timeout_s) as conn:
            user = self._get(conn, user_id)
            if user is None:
                raise PipelineError(ErrorCode.NOT_FOUND, "This user does not exist.")
            if (
                _is_active_admin(user.role, user.is_active)
                and self._active_admins(conn, excluding=user_id) == 0
            ):
                raise PipelineError(ErrorCode.LAST_ADMIN)
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            job_ids = delete_jobs_of_owner(conn, user_id)
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return job_ids

    def has_active_admin(self) -> bool:
        with db.connection(self.db_path, timeout_s=self._timeout_s) as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE role = 'admin' AND is_active = 1 LIMIT 1"
            ).fetchone()
        return row is not None
