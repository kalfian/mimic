"""In-memory login throttling (PLAN-auth A6)."""

from __future__ import annotations

import pytest

from app.auth.ratelimit import LoginLimiter


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def limiter(clock: Clock) -> LoginLimiter:
    return LoginLimiter(window_s=900, max_per_key=5, max_per_ip=30, clock=clock)


def test_fifth_failure_blocks_the_key(limiter: LoginLimiter, clock: Clock) -> None:
    for _ in range(4):
        limiter.record_failure("1.1.1.1", "alice")
        assert limiter.retry_after("1.1.1.1", "alice") is None
    limiter.record_failure("1.1.1.1", "alice")
    assert limiter.retry_after("1.1.1.1", "alice") == 900
    # other username from the same IP, or same username from another IP: not blocked
    assert limiter.retry_after("1.1.1.1", "bob") is None
    assert limiter.retry_after("2.2.2.2", "alice") is None


def test_sliding_window(limiter: LoginLimiter, clock: Clock) -> None:
    for i in range(5):
        clock.t = 1000.0 + i * 100
        limiter.record_failure("ip", "alice")
    clock.t = 1500.0
    # oldest failure (t=1000) leaves the window at t=1900
    assert limiter.retry_after("ip", "alice") == 400
    clock.t = 1899.5
    assert limiter.retry_after("ip", "alice") == 1  # never 0 while blocked
    clock.t = 1900.0
    assert limiter.retry_after("ip", "alice") is None


def test_per_ip_limit(limiter: LoginLimiter, clock: Clock) -> None:
    for i in range(30):
        limiter.record_failure("ip", f"user{i}")
    assert limiter.retry_after("ip", "fresh-name") == 900
    assert limiter.retry_after("other-ip", "fresh-name") is None


def test_reset_clears_key_but_not_ip(limiter: LoginLimiter) -> None:
    for _ in range(5):
        limiter.record_failure("ip", "alice")
    limiter.reset("ip", "alice")
    assert limiter.retry_after("ip", "alice") is None
    for i in range(25):
        limiter.record_failure("ip", f"u{i}")
    # 5 (reset key, still counted per IP) + 25 = 30 -> IP blocked
    assert limiter.retry_after("ip", "alice") is not None


def test_entries_expire(limiter: LoginLimiter, clock: Clock) -> None:
    limiter.record_failure("ip", "alice")
    clock.t += 901
    assert limiter.retry_after("ip", "alice") is None
    assert limiter._by_key == {} and limiter._by_ip == {}


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        LoginLimiter(max_per_key=0)
