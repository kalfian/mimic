"""scrypt password hashing (PLAN-auth A3, A4, A5)."""

from __future__ import annotations

import base64
import hashlib

import pytest

from app.auth import passwords, policy

PW = "correct horse battery staple"


def test_round_trip_and_wrong_password() -> None:
    encoded = passwords.hash_password(PW)
    assert passwords.verify_password(PW, encoded)
    assert not passwords.verify_password(PW + "!", encoded)
    assert not passwords.verify_password("", encoded)


def test_encoding_carries_params() -> None:
    encoded = passwords.hash_password(PW)
    scheme, log2_n, r, p, salt, digest = encoded.split("$")
    params = policy.current_scrypt_params()
    assert (scheme, int(log2_n), int(r), int(p)) == ("scrypt", params.log2_n, params.r, params.p)
    assert len(base64.b64decode(salt)) == passwords.SALT_BYTES
    assert len(base64.b64decode(digest)) == params.dklen
    assert PW not in encoded


def test_unique_salts() -> None:
    a, b = passwords.hash_password(PW), passwords.hash_password(PW)
    assert a != b and a.split("$")[4] != b.split("$")[4]


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "!unusable",
        "scrypt$10$8$1$abc",  # too few fields
        "bcrypt$10$8$1$AAAA$AAAA",
        "scrypt$x$8$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 44,
        "scrypt$10$8$1$not base64!$" + "A" * 44,
        "scrypt$40$8$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 44,  # absurd cost
        "scrypt$10$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAA",  # digest too short
    ],
)
def test_malformed_hash_is_false_not_an_exception(encoded: str) -> None:
    assert passwords.verify_password(PW, encoded) is False
    assert passwords.needs_rehash(encoded) is True


def test_needs_rehash_follows_current_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = passwords.hash_password(PW)
    assert not passwords.needs_rehash(encoded)
    monkeypatch.setattr(policy, "SCRYPT_PARAMS", policy.ScryptParams(log2_n=11))
    assert passwords.needs_rehash(encoded)
    # old hashes still verify: the stored params are used, not the current ones
    assert passwords.verify_password(PW, encoded)


def test_cost_is_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "SCRYPT_PARAMS", policy.ScryptParams(log2_n=9))
    assert passwords.hash_password(PW).split("$")[1] == "9"


def test_compare_digest_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = passwords.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(passwords.hmac, "compare_digest", spy)
    encoded = passwords.hash_password(PW)
    passwords.verify_password(PW, encoded)
    passwords.verify_password("nope nope nope", encoded)
    assert len(calls) == 2


def test_nfkc_normalization() -> None:
    # U+FB01 (ligature) and "fi" hash the same after NFKC
    encoded = passwords.hash_password("ﬁne passphrase")
    assert passwords.verify_password("fine passphrase", encoded)


def test_maxmem_is_passed_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}
    real = hashlib.scrypt

    def spy(password: bytes, **kw: int) -> bytes:
        seen.update(kw)
        return real(password, **kw)

    monkeypatch.setattr(passwords.hashlib, "scrypt", spy)
    passwords.hash_password(PW)
    assert seen["maxmem"] >= policy.current_scrypt_params().maxmem


@pytest.mark.real_scrypt
def test_production_cost_works_with_explicit_maxmem() -> None:
    # N = 2^15, r = 8: 32 MiB, which fails under OpenSSL's default cap without maxmem.
    encoded = passwords.hash_password(PW)
    assert encoded.startswith("scrypt$15$8$1$")
    assert passwords.verify_password(PW, encoded)


def test_dummy_verify_runs_without_error() -> None:
    passwords.dummy_verify(PW)
    passwords.dummy_verify("")


def test_temporary_password_shape_and_policy() -> None:
    temps = {passwords.generate_temporary_password() for _ in range(20)}
    assert len(temps) == 20
    for t in temps:
        assert len(t) == 16
        assert policy.check_password_policy(t, "someone") is None
