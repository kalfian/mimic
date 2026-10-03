/**
 * Rounding parity with the backend generators (api/app/generate/phrasing.py: Decimal ROUND_HALF_UP
 * on the shortest repr). Expected strings were produced by phrasing.px / scale_num / opacity_num
 * / ms_int, so the UI and the generated spec text never disagree on a rounded number.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { newPasswordProblems, passwordLength, usernameProblem } from "./authPolicy";
import { ApiError, parseRetryAfter, type ApiErrorCode } from "./errors";
import {
  describeApiError,
  describeError,
  formatJobOwner,
  formatLastLogin,
  formatMs,
  formatOpacity,
  formatPx,
  formatRelativeTime,
  formatRetryAfter,
  formatScale,
  SETUP_COMMAND,
  USER_ROLE_LABELS,
} from "./format";

test("formatPx rounds half away from zero like the backend", () => {
  const cases: [number, string][] = [
    [2.25, "2.5px"], [-2.25, "-2.5px"], [3.5, "4px"], [-3.5, "-4px"], [-8, "-8px"],
    [0.24, "0px"], [-0.26, "-0.5px"], [7.4999, "7px"],
  ];
  for (const [v, want] of cases) assert.equal(formatPx(v), want, `formatPx(${v})`);
});

test("formatScale / formatOpacity / formatMs match the backend", () => {
  for (const [v, want] of [[1.005, "1.01"], [0.995, "1.00"], [1.06, "1.06"], [0.955, "0.96"]] as const) {
    assert.equal(formatScale(v), want, `formatScale(${v})`);
  }
  for (const [v, want] of [[0.425, "0.45"], [0.475, "0.5"], [0.97, "0.95"]] as const) {
    assert.equal(formatOpacity(v), want, `formatOpacity(${v})`);
  }
  for (const [v, want] of [[275, "~280 ms"], [285, "~290 ms"], [-15, "~-20 ms"]] as const) {
    assert.equal(formatMs(v, { approx: true }), want, `formatMs(${v})`);
  }
});

/* ---------- auth / admin (PLAN-auth §8.2) ---------- */

const ALL_CODES: ApiErrorCode[] = [
  "unsupported_format", "file_too_large", "too_long", "too_short", "decode_failed", "invalid_request",
  "no_motion_detected", "unsupported_motion", "no_stable_state", "internal_error", "interrupted",
  "not_found", "not_ready", "already_running", "unauthenticated", "invalid_credentials", "account_disabled",
  "forbidden", "password_change_required", "origin_not_allowed", "too_many_attempts", "setup_required",
  "last_admin", "self_action_forbidden", "username_taken", "weak_password", "current_password_incorrect",
  "network_error", "aborted", "invalid_response", "cookie_rejected",
];

test("describeError covers every code", () => {
  for (const code of ALL_CODES) {
    const d = describeError(code);
    assert.ok(d.title.length > 0 && d.message.length > 0, code);
  }
  assert.ok(describeError("setup_required").guidance.some((g) => g.includes(SETUP_COMMAND)));
  assert.match(describeError("cookie_rejected").guidance.join(" "), /same hostname/);
  assert.match(describeError("not_found").message, /don't have access/);
});

test("describeApiError: server message, client message, Retry-After", () => {
  const weak = ApiError.fromResponse(422, { error: { code: "weak_password", message: "Use at least 12 characters." } });
  assert.equal(describeApiError(weak).message, "Use at least 12 characters.");
  const bare = ApiError.fromResponse(401, null);
  assert.equal(bare.code, "unauthenticated");
  assert.equal(bare.serverMessage, null);
  assert.equal(describeApiError(bare).message, describeError("unauthenticated").message);
  const limited = ApiError.fromResponse(429, { error: { code: "too_many_attempts", message: "Too many." } }, { retryAfter: "120" });
  assert.deepEqual(describeApiError(limited).guidance, ["Try again in 2 min."]);
  const cookie = ApiError.cookieRejected({ pageHost: "127.0.0.1", apiHost: "localhost" });
  assert.match(describeApiError(cookie).message, /127\.0\.0\.1/);
});

test("describeApiError: network_error on a mismatched hostname names the cause first", () => {
  const net = ApiError.network();
  const plain = describeApiError(net);
  assert.deepEqual(describeApiError(net, { hostnameMismatch: null }), plain);
  const d = describeApiError(net, { hostnameMismatch: { pageHost: "127.0.0.1", apiHost: "localhost" } });
  assert.equal(d.message, plain.message);
  assert.match(d.guidance[0], /This page is on 127\.0\.0\.1 but the API is on localhost/);
  assert.match(d.guidance[1], /Open the app on localhost/);
  assert.deepEqual(d.guidance.slice(2), plain.guidance);
  // Only network errors: other codes ignore the context.
  const forbidden = ApiError.fromResponse(403, { error: { code: "forbidden", message: "No." } });
  assert.deepEqual(describeApiError(forbidden, { hostnameMismatch: { pageHost: "a", apiHost: "b" } }), describeApiError(forbidden));
});

test("describeApiError: advice appears once (message vs guidance)", () => {
  // cookie_rejected: the message says what happened, the guidance says what to do, no overlap.
  const cookie = describeApiError(ApiError.cookieRejected({ pageHost: "127.0.0.1", apiHost: "localhost" }));
  assert.doesNotMatch(cookie.message, /same hostname/);
  assert.deepEqual(cookie.guidance, describeError("cookie_rejected").guidance);
  assert.ok(cookie.guidance.every((g) => !cookie.message.includes(g.replace(/\.$/, ""))));
  // A server message that already ends with the guidance drops that guidance line.
  const last = ApiError.fromResponse(409, {
    error: { code: "last_admin", message: "This is the last active admin. Make another user an admin first." },
  });
  assert.deepEqual(describeApiError(last).guidance, []);
  // Unrelated guidance stays.
  const taken = ApiError.fromResponse(409, { error: { code: "username_taken", message: "That username is already taken." } });
  assert.deepEqual(describeApiError(taken).guidance, ["Choose another username."]);
});

test("ApiError: Retry-After parsing and transience", () => {
  assert.equal(parseRetryAfter("90"), 90);
  assert.equal(parseRetryAfter(null), null);
  assert.equal(parseRetryAfter("soon"), null);
  const now = Date.parse("2026-10-03T10:00:00Z");
  assert.equal(parseRetryAfter("Sat, 03 Oct 2026 10:01:30 GMT", now), 90);
  assert.equal(parseRetryAfter("Sat, 03 Oct 2026 09:00:00 GMT", now), 0);
  assert.equal(ApiError.fromResponse(429, null).isTransient, false);
  assert.equal(ApiError.fromResponse(429, null).code, "too_many_attempts");
  assert.equal(ApiError.fromResponse(503, null).isTransient, true);
  assert.equal(ApiError.fromResponse(401, { error: { code: "unauthenticated", message: "x" } }).isTransient, false);
  assert.equal(ApiError.cookieRejected().isTransient, false);
});

test("account formatting", () => {
  assert.deepEqual(USER_ROLE_LABELS, { admin: "Admin", user: "User" });
  assert.equal(formatRetryAfter(0), "Try again now");
  assert.equal(formatRetryAfter(45), "Try again in 45 s");
  assert.equal(formatRetryAfter(61), "Try again in 2 min");
  assert.equal(formatRetryAfter(900), "Try again in 15 min");
  assert.equal(formatRetryAfter(3900), "Try again in 1 h 5 min");
  const now = Date.parse("2026-10-03T12:00:00Z");
  assert.equal(formatLastLogin(null, now), "Never");
  assert.equal(formatRelativeTime("2026-10-03T11:59:30Z", now), "just now");
  assert.equal(formatRelativeTime("2026-10-03T11:55:00Z", now), "5 min ago");
  assert.equal(formatRelativeTime("2026-10-03T09:00:00Z", now), "3 h ago");
  assert.equal(formatLastLogin("2026-10-02T10:00:00Z", now), "yesterday");
  assert.equal(formatRelativeTime("2026-09-29T12:00:00Z", now), "4 days ago");
  assert.match(formatRelativeTime("2026-09-01T12:00:00Z", now), /2026/);
  assert.equal(formatRelativeTime("garbage", now), "—");
  assert.equal(formatJobOwner(null), "Legacy");
  assert.equal(formatJobOwner({ id: "x", username: "ann" }), "ann");
});

test("client-side account validation mirrors the server policy", () => {
  assert.equal(usernameProblem(" Alice "), null);
  assert.match(usernameProblem("ab") ?? "", /3–32/);
  assert.match(usernameProblem("") ?? "", /Enter/);
  assert.equal(passwordLength("😀".repeat(12)), 12); // code points, not UTF-16 units
  assert.equal(passwordLength("ﬁ"), 2); // NFKC: the ligature becomes "fi"
  assert.deepEqual(newPasswordProblems({ newPassword: "a long passphrase", confirmPassword: "a long passphrase", username: "ann" }), {});
  const p = newPasswordProblems({ newPassword: "short", confirmPassword: "other", username: "ann" });
  assert.match(p.newPassword ?? "", /at least 12/);
  assert.match(p.confirmPassword ?? "", /match/);
  assert.match(newPasswordProblems({ newPassword: "Annannannann", confirmPassword: "Annannannann", username: "annannannann" }).newPassword ?? "", /username/);
  assert.match(
    newPasswordProblems({ newPassword: "same old passphrase", confirmPassword: "same old passphrase", username: "ann", currentPassword: "same old passphrase" }).newPassword ?? "",
    /different/,
  );
});
