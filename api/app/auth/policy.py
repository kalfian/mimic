"""Username / password policy and job visibility rules (PLAN-auth A3, A4, A10, A17, U3).

Single source of truth for the limits mirrored to ``web/lib/types.ts`` (``make contract``).
Pure functions only: no I/O, no imports from the rest of ``app`` except stdlib.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.auth.models import UserRecord

#: Usernames: lowercase ASCII letters/digits, then ``.``, ``_`` or ``-``; 3–32 chars (A17).
USERNAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,31}$"
USERNAME_RE = re.compile(USERNAME_PATTERN)

#: Password length bounds, counted in characters after NFKC normalization (A4).
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 256

#: User ids are ``uuid4().hex`` (same shape as job ids).
USER_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def normalize_username(raw: str) -> str:
    """Strip + lowercase. Does not validate (see :func:`is_valid_username`)."""
    return raw.strip().lower()


def is_valid_username(username: str) -> bool:
    """True if an already-normalized username matches :data:`USERNAME_PATTERN`."""
    return USERNAME_RE.fullmatch(username) is not None


def is_valid_user_id(user_id: str) -> bool:
    return USER_ID_RE.fullmatch(user_id) is not None


def normalize_password(password: str) -> str:
    """NFKC form. Hash and verify this, never the raw input, so equivalent input matches."""
    return unicodedata.normalize("NFKC", password)


def check_password_policy(
    password: str, username: str, *, current_password: str | None = None
) -> str | None:
    """Violation message for a *new* password, or None if it is acceptable.

    ``password`` and ``current_password`` are raw input (normalized here). ``username`` is the
    account's normalized username. The message is user-facing (``weak_password``) and never
    echoes the password.
    """
    pw = normalize_password(password)
    if len(pw) < PASSWORD_MIN_LENGTH:
        return f"Use at least {PASSWORD_MIN_LENGTH} characters."
    if len(pw) > PASSWORD_MAX_LENGTH:
        return f"Use at most {PASSWORD_MAX_LENGTH} characters."
    if pw.casefold() == username.casefold():
        return "The password must not be the same as the username."
    if current_password is not None and pw == normalize_password(current_password):
        return "The new password must be different from the current one."
    return None


# ---- scrypt cost (A3) ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScryptParams:
    """Cost parameters for *new* hashes. Stored hashes encode their own parameters."""

    log2_n: int = 15
    r: int = 8
    p: int = 1
    dklen: int = 32
    #: Passed explicitly: N·r·128 = 32 MiB is at OpenSSL's default cap and raises without it.
    maxmem: int = 64 * 1024 * 1024


#: Current cost. Tests lower it via the autouse fixture in ``tests/conftest.py``; read it through
#: :func:`current_scrypt_params` at call time (never copy it into a module-level constant).
SCRYPT_PARAMS = ScryptParams()


def current_scrypt_params() -> ScryptParams:
    return SCRYPT_PARAMS


# ---- job visibility (U3, A10) ----------------------------------------------------------------


def can_access_job(user: UserRecord, owner_id: str | None) -> bool:
    """Admins see every job; users only their own. ``owner_id is None`` (legacy) = admin-only.

    Callers turn ``False`` into 404 ``not_found`` (never 403), so existence does not leak.
    """
    if user.is_admin:
        return True
    return owner_id is not None and owner_id == user.id
