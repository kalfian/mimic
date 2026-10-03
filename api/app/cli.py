"""Operator CLI (PLAN-auth §6): ``uv run python -m app.cli <command>`` (from ``api/``).

Commands
--------
``create-admin [--username NAME] [--password-stdin]``
    Migrate the database, create an active admin. The first admin claims every legacy
    (ownerless) job (A10). Exit 0 ok, 2 invalid input, 3 username taken.
``reset-password USERNAME [--password-stdin]``
    Break-glass: set a new password (no forced change), enable the account, revoke its
    sessions. Exit 0 ok, 2 invalid input / unknown user.
``list-users``
    Username, role, active, must-change, job count. Never a hash.
``clean-jobs --yes``
    Delete every job row and ``<data>/jobs/*``; keep users and sessions (A16). Without
    ``--yes`` it only says what it would do and exits 1.

Passwords come from an interactive ``getpass`` prompt (twice, up to 3 tries) or, with
``--password-stdin``, from one line of stdin (scripts/tests). Never from argv. Nothing here
prints a password or hash. The data dir is ``MIMIC_DATA_DIR`` (``app.config``).
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import sys
from collections.abc import Sequence
from typing import TextIO

from app.auth import passwords, policy
from app.auth.sessions import SqliteSessionStore
from app.auth.users import SqliteUserStore
from app.config import Settings, get_settings
from app.core import db
from app.core.errors import ErrorCode, PipelineError
from app.core.storage import is_valid_job_id

EXIT_OK = 0
EXIT_ABORTED = 1
EXIT_INVALID = 2
EXIT_TAKEN = 3
PASSWORD_TRIES = 3


class _Abort(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _migrate(settings: Settings) -> None:
    result = db.migrate(settings.db_path)
    if result.backup_path is not None:
        print(f"Backed up the existing database to {result.backup_path}")
    if result.migrated:
        print(f"Migrated {settings.db_path} to schema v{result.to_version}")


def _read_stdin_password(stdin: TextIO) -> str:
    line = stdin.readline()
    if not line:
        raise _Abort(EXIT_INVALID, "no password on stdin")
    return line.rstrip("\r\n")


def _new_password(username: str, *, from_stdin: bool, stdin: TextIO) -> str:
    """A policy-checked new password (A4) for ``username``."""
    if from_stdin:
        password = _read_stdin_password(stdin)
        violation = policy.check_password_policy(password, username)
        if violation is not None:
            raise _Abort(EXIT_INVALID, f"weak password: {violation}")
        return password
    for _ in range(PASSWORD_TRIES):
        password = getpass.getpass("Password: ")
        violation = policy.check_password_policy(password, username)
        if violation is not None:
            _err(f"weak password: {violation}")
            continue
        if getpass.getpass("Repeat password: ") != password:
            _err("passwords do not match")
            continue
        return password
    raise _Abort(EXIT_INVALID, f"no valid password after {PASSWORD_TRIES} tries")


# ---- commands --------------------------------------------------------------------------------


def cmd_create_admin(args: argparse.Namespace, settings: Settings, stdin: TextIO) -> int:
    _migrate(settings)
    users = SqliteUserStore(settings.db_path)
    raw = args.username if args.username is not None else input("Username: ")
    username = policy.normalize_username(raw)
    if not policy.is_valid_username(username):
        raise _Abort(
            EXIT_INVALID,
            "invalid username: use 3-32 characters, lowercase letters, digits, '.', '_' or '-' "
            "(starting with a letter or digit)",
        )
    if users.get_by_username(username) is not None:
        raise _Abort(EXIT_TAKEN, f"username {username!r} is already taken")
    password = _new_password(username, from_stdin=args.password_stdin, stdin=stdin)
    try:
        admin, claimed = users.create_admin(username, passwords.hash_password(password))
    except PipelineError as exc:
        if exc.code is ErrorCode.USERNAME_TAKEN:
            raise _Abort(EXIT_TAKEN, f"username {username!r} is already taken") from None
        raise _Abort(EXIT_INVALID, exc.message) from None
    print(f"Created admin {admin.username}.")
    if claimed is None:
        print("Note: another active admin already exists; no existing jobs were reassigned.")
    else:
        print(f"Assigned {claimed} existing jobs to {admin.username}.")
    return EXIT_OK


def cmd_reset_password(args: argparse.Namespace, settings: Settings, stdin: TextIO) -> int:
    _migrate(settings)
    users = SqliteUserStore(settings.db_path)
    user = users.get_by_username(args.username)
    if user is None:
        raise _Abort(EXIT_INVALID, f"no user named {policy.normalize_username(args.username)!r}")
    password = _new_password(user.username, from_stdin=args.password_stdin, stdin=stdin)
    if not user.is_active:
        users.update(user.id, is_active=True)
    users.set_password(user.id, passwords.hash_password(password), must_change=False)
    revoked = SqliteSessionStore.from_settings(settings).revoke_all(user.id)
    enabled = "" if user.is_active else " The account was enabled."
    print(f"Password of {user.username} reset; {revoked} sessions revoked.{enabled}")
    return EXIT_OK


def cmd_list_users(args: argparse.Namespace, settings: Settings, stdin: TextIO) -> int:
    _migrate(settings)
    rows = SqliteUserStore(settings.db_path).list_with_job_counts()
    if not rows:
        print("No users. Create the first admin with `make create-admin`.")
        return EXIT_OK
    header = ("USERNAME", "ROLE", "ACTIVE", "MUST-CHANGE", "JOBS")
    table = [
        (
            u.username,
            u.role.value,
            "yes" if u.is_active else "no",
            "yes" if u.must_change_password else "no",
            str(n),
        )
        for u, n in rows
    ]
    widths = [max(len(r[i]) for r in (header, *table)) for i in range(len(header))]
    for row in (header, *table):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
    return EXIT_OK


def cmd_clean_jobs(args: argparse.Namespace, settings: Settings, stdin: TextIO) -> int:
    _migrate(settings)
    jobs_dir = settings.jobs_dir
    with db.connection(settings.db_path) as conn:
        count = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    if not args.yes:
        print(
            f"Would delete {count} job rows and everything under {jobs_dir}. Users and sessions "
            "are kept. Re-run with --yes to do it."
        )
        return EXIT_ABORTED
    with db.transaction(settings.db_path) as conn:
        deleted = conn.execute("DELETE FROM jobs").rowcount
    removed = 0
    if jobs_dir.is_dir():
        for entry in jobs_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += is_valid_job_id(entry.name)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Deleted {deleted} job rows and {removed} job directories. Users were kept.")
    return EXIT_OK


# ---- entry point -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Mimic operator CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-admin", help="create an admin account (first one claims old jobs)")
    p.add_argument("--username", help="prompted for when omitted")
    p.add_argument(
        "--password-stdin", action="store_true", help="read the password from one stdin line"
    )
    p.set_defaults(func=cmd_create_admin)

    p = sub.add_parser("reset-password", help="set a new password, enable, revoke sessions")
    p.add_argument("username")
    p.add_argument(
        "--password-stdin", action="store_true", help="read the password from one stdin line"
    )
    p.set_defaults(func=cmd_reset_password)

    p = sub.add_parser("list-users", help="list accounts (no hashes)")
    p.set_defaults(func=cmd_list_users)

    p = sub.add_parser("clean-jobs", help="delete all jobs, keep accounts")
    p.add_argument("--yes", action="store_true", help="actually delete")
    p.set_defaults(func=cmd_clean_jobs)
    return parser


def main(argv: Sequence[str] | None = None, *, stdin: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        return int(args.func(args, settings, stdin if stdin is not None else sys.stdin))
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except (KeyboardInterrupt, EOFError):
        print(file=sys.stderr)
        _err("aborted")
        return EXIT_ABORTED
    except RuntimeError as exc:  # e.g. database schema newer than this code
        _err(str(exc))
        return EXIT_ABORTED


if __name__ == "__main__":
    sys.exit(main())
