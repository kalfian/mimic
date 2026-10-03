"""Operator CLI (PLAN-auth §6, A10, A16)."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import cli
from app.api.schemas import JobOptions
from app.auth.models import Role
from app.auth.passwords import verify_password
from app.auth.sessions import SqliteSessionStore
from app.auth.users import SqliteUserStore
from app.config import Settings
from app.core import db
from app.core.jobstore import SqliteJobStore

PW = "operator passphrase 1"
PW2 = "operator passphrase 2"


@pytest.fixture
def db_path(settings: Settings) -> Path:
    return settings.db_path


def run(*argv: str, stdin: str = "") -> int:
    return cli.main(list(argv), stdin=io.StringIO(stdin))


def _job(db_path: Path, job_id: str, owner: str | None = None) -> None:
    SqliteJobStore(db_path).create(
        job_id, original_filename="a.mp4", ext="mp4", options=JobOptions(), owner_id=owner
    )


@pytest.fixture
def fake_getpass(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    answers: list[str] = []

    def getpass(prompt: str = "") -> str:
        if not answers:
            raise EOFError
        return answers.pop(0)

    monkeypatch.setattr(cli.getpass, "getpass", getpass)
    yield answers


# ---- create-admin --------------------------------------------------------------------------


def test_create_admin_stdin_claims_orphans(
    db_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db.migrate(db_path)
    _job(db_path, "a" * 32)
    _job(db_path, "b" * 32)
    assert run("create-admin", "--username", " Root ", "--password-stdin", stdin=PW + "\n") == 0
    out = capsys.readouterr().out
    assert "Assigned 2 existing jobs to root" in out
    assert PW not in out
    admin = SqliteUserStore(db_path).get_by_username("root")
    assert admin is not None and admin.role is Role.ADMIN and admin.is_active
    assert not admin.must_change_password and verify_password(PW, admin.password_hash)
    rec = SqliteJobStore(db_path).get("a" * 32)
    assert rec is not None and rec.owner_id == admin.id


def test_second_admin_does_not_claim(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("create-admin", "--username", "root", "--password-stdin", stdin=PW)
    _job(db_path, "c" * 32)
    capsys.readouterr()
    assert run("create-admin", "--username", "root2", "--password-stdin", stdin=PW) == 0
    assert "another active admin already exists" in capsys.readouterr().out
    rec = SqliteJobStore(db_path).get("c" * 32)
    assert rec is not None and rec.owner_id is None


def test_create_admin_interactive(
    db_path: Path, fake_getpass: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "root")
    # weak, then mismatch, then ok
    fake_getpass.extend(["short", PW, PW2, PW, PW])
    assert run("create-admin") == 0
    admin = SqliteUserStore(db_path).get_by_username("root")
    assert admin is not None and verify_password(PW, admin.password_hash)


def test_create_admin_gives_up_after_three_tries(db_path: Path, fake_getpass: list[str]) -> None:
    fake_getpass.extend(["short", "short", "short"])
    assert run("create-admin", "--username", "root") == cli.EXIT_INVALID
    assert SqliteUserStore(db_path).get_by_username("root") is None


@pytest.mark.parametrize(
    ("argv", "stdin", "code"),
    [
        (("--username", "ab", "--password-stdin"), PW, cli.EXIT_INVALID),
        (("--username", "root", "--password-stdin"), "short", cli.EXIT_INVALID),
        (("--username", "rootroot1234", "--password-stdin"), "ROOTROOT1234", cli.EXIT_INVALID),
        (("--username", "root", "--password-stdin"), "", cli.EXIT_INVALID),
    ],
)
def test_create_admin_invalid_input(
    db_path: Path, argv: tuple[str, ...], stdin: str, code: int
) -> None:
    assert run("create-admin", *argv, stdin=stdin) == code
    db.migrate(db_path)
    assert not SqliteUserStore(db_path).has_active_admin()


def test_create_admin_username_taken(db_path: Path) -> None:
    assert run("create-admin", "--username", "root", "--password-stdin", stdin=PW) == 0
    assert run("create-admin", "--username", "ROOT", "--password-stdin", stdin=PW) == cli.EXIT_TAKEN


def test_no_password_option_in_argv() -> None:
    parser = cli.build_parser()
    for cmd in (["create-admin"], ["reset-password", "root"]):
        with pytest.raises(SystemExit):
            parser.parse_args([*cmd, "--password", PW])
    help_text = parser.format_help() + "".join(
        a.format_help()
        for a in parser._subparsers._group_actions[0].choices.values()  # type: ignore[union-attr]
    )
    assert "--password " not in help_text and "--password=" not in help_text


# ---- reset-password ------------------------------------------------------------------------


def test_reset_password_reenables_and_revokes(
    db_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run("create-admin", "--username", "root", "--password-stdin", stdin=PW)
    users = SqliteUserStore(db_path)
    root = users.get_by_username("root")
    assert root is not None
    alice = users.create("alice", "!unusable", Role.USER, must_change_password=True)
    sessions = SqliteSessionStore.from_settings(settings)
    token = sessions.create(alice.id)
    users.update(alice.id, is_active=False)  # the store leaves the session row in place
    capsys.readouterr()

    assert run("reset-password", "Alice", "--password-stdin", stdin=PW2) == 0
    out = capsys.readouterr().out
    assert "1 sessions revoked" in out and "enabled" in out and PW2 not in out
    after = users.get(alice.id)
    assert after is not None and after.is_active and not after.must_change_password
    assert verify_password(PW2, after.password_hash)
    assert sessions.resolve(token) is None


def test_reset_password_unknown_user_and_weak(db_path: Path) -> None:
    assert run("reset-password", "nobody", "--password-stdin", stdin=PW) == cli.EXIT_INVALID
    run("create-admin", "--username", "root", "--password-stdin", stdin=PW)
    assert run("reset-password", "root", "--password-stdin", stdin="short") == cli.EXIT_INVALID
    root = SqliteUserStore(db_path).get_by_username("root")
    assert root is not None and verify_password(PW, root.password_hash)


# ---- list-users / clean-jobs ---------------------------------------------------------------


def test_list_users(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("list-users") == 0
    assert "No users" in capsys.readouterr().out
    run("create-admin", "--username", "root", "--password-stdin", stdin=PW)
    root = SqliteUserStore(db_path).get_by_username("root")
    assert root is not None
    _job(db_path, "a" * 32, root.id)
    capsys.readouterr()
    assert run("list-users") == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["USERNAME", "ROLE", "ACTIVE", "MUST-CHANGE", "JOBS"]
    assert out.splitlines()[1].split() == ["root", "admin", "yes", "no", "1"]
    assert "scrypt" not in out


def test_clean_jobs_keeps_users(
    db_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run("create-admin", "--username", "root", "--password-stdin", stdin=PW)
    _job(db_path, "a" * 32)
    job_dir = settings.jobs_dir / ("a" * 32)
    job_dir.mkdir(parents=True)
    (job_dir / "input.mp4").write_bytes(b"x")
    capsys.readouterr()

    assert run("clean-jobs") == cli.EXIT_ABORTED  # dry run
    assert "Would delete 1 job rows" in capsys.readouterr().out
    assert job_dir.is_dir() and SqliteJobStore(db_path).get("a" * 32) is not None

    assert run("clean-jobs", "--yes") == 0
    assert "Deleted 1 job rows and 1 job directories" in capsys.readouterr().out
    assert not job_dir.exists() and settings.jobs_dir.is_dir()
    assert SqliteJobStore(db_path).get("a" * 32) is None
    assert SqliteUserStore(db_path).has_active_admin()


def test_newer_schema_is_reported(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db.migrate(db_path)
    with db.connection(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    assert run("list-users") == cli.EXIT_ABORTED
    assert "newer than this code" in capsys.readouterr().err
