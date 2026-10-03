"""Password hashing with stdlib ``hashlib.scrypt`` (PLAN-auth A3, A5).

Encoding: ``scrypt$<log2N>$<r>$<p>$<salt b64>$<hash b64>`` (standard base64, padded). The
parameters travel with the hash, so verification uses the stored cost and the cost for *new*
hashes can be raised later (:func:`needs_rehash` tells the login route to upgrade).

* The cost is read from :func:`app.auth.policy.current_scrypt_params` on every call, never
  copied into a module constant (tests lower it through ``policy.SCRYPT_PARAMS``).
* ``maxmem`` is always passed explicitly: N·r·128 = 32 MiB at the production cost sits on
  OpenSSL's default 32 MiB cap, which raises ``ValueError`` without it.
* At most :data:`MAX_CONCURRENT_HASHES` derivations run at once (memory cap under a burst).
* Passwords are NFKC-normalized before hashing (:func:`app.auth.policy.normalize_password`).
* Nothing here logs or returns a password, salt or derived key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass

from app.auth import policy

SCHEME = "scrypt"
SALT_BYTES = 16
MAX_CONCURRENT_HASHES = 4

#: Upper bounds accepted when *parsing* a stored hash, so a corrupted or hostile value can't
#: make verification allocate gigabytes (N·r·128 ≤ 256 MiB, p ≤ 16).
_MAX_LOG2_N = 20
_MAX_R = 32
_MAX_P = 16
_MAX_MEM_BYTES = 256 * 1024 * 1024
_MIN_DKLEN, _MAX_DKLEN = 16, 64

_hash_slots = threading.BoundedSemaphore(MAX_CONCURRENT_HASHES)

#: Dummy hashes for unknown usernames, one per cost (the cost only changes in tests).
_dummy_hashes: dict[policy.ScryptParams, str] = {}
_dummy_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class _Parsed:
    log2_n: int
    r: int
    p: int
    salt: bytes
    digest: bytes


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def _derive(password: str, salt: bytes, *, log2_n: int, r: int, p: int, dklen: int) -> bytes:
    n = 1 << log2_n
    # Explicit maxmem: the configured cap, or more if a stored hash needs it (bounded by the
    # parse limits above). 2x the block memory leaves room for OpenSSL's own overhead.
    maxmem = max(policy.current_scrypt_params().maxmem, 2 * 128 * n * r * p)
    with _hash_slots:
        return hashlib.scrypt(
            policy.normalize_password(password).encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=dklen,
            maxmem=maxmem,
        )


def _parse(encoded: str) -> _Parsed | None:
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != SCHEME:
        return None
    try:
        log2_n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, digest = _unb64(parts[4]), _unb64(parts[5])
    except (ValueError, UnicodeError, binascii.Error):
        return None
    if not (1 <= log2_n <= _MAX_LOG2_N and 1 <= r <= _MAX_R and 1 <= p <= _MAX_P):
        return None
    if 128 * (1 << log2_n) * r > _MAX_MEM_BYTES:
        return None
    if not salt or not (_MIN_DKLEN <= len(digest) <= _MAX_DKLEN):
        return None
    return _Parsed(log2_n=log2_n, r=r, p=p, salt=salt, digest=digest)


def hash_password(password: str) -> str:
    """New salted hash of ``password`` at the current cost."""
    params = policy.current_scrypt_params()
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(
        password, salt, log2_n=params.log2_n, r=params.r, p=params.p, dklen=params.dklen
    )
    return f"{SCHEME}${params.log2_n}${params.r}${params.p}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """True if ``password`` matches ``encoded``. A malformed hash is False, never an exception."""
    parsed = _parse(encoded)
    if parsed is None:
        return False
    try:
        candidate = _derive(
            password,
            parsed.salt,
            log2_n=parsed.log2_n,
            r=parsed.r,
            p=parsed.p,
            dklen=len(parsed.digest),
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, parsed.digest)


def needs_rehash(encoded: str) -> bool:
    """True if ``encoded`` was not produced with the current cost (or is malformed)."""
    parsed = _parse(encoded)
    if parsed is None:
        return True
    params = policy.current_scrypt_params()
    return (parsed.log2_n, parsed.r, parsed.p, len(parsed.digest)) != (
        params.log2_n,
        params.r,
        params.p,
        params.dklen,
    )


def dummy_verify(password: str) -> None:
    """Spend one verification at the current cost and discard the result.

    Used for unknown usernames so a login takes the same time whether or not the account
    exists (A6, enumeration resistance).
    """
    params = policy.current_scrypt_params()
    with _dummy_lock:
        encoded = _dummy_hashes.get(params)
    if encoded is None:
        encoded = hash_password(secrets.token_urlsafe(16))
        with _dummy_lock:
            _dummy_hashes.setdefault(params, encoded)
    verify_password(password, encoded)


def generate_temporary_password() -> str:
    """Server-generated temporary password (A5): 16 URL-safe characters, shown once."""
    return secrets.token_urlsafe(12)
