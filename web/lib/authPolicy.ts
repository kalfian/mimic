/**
 * Client-side mirror of the account rules in api/app/auth/policy.py (PLAN-auth A4, A17), for
 * instant form feedback. The server stays authoritative: it re-checks everything and answers
 * `weak_password` / `invalid_request` with the specific rule in `message`. Pure, no React.
 */

import { PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH, USERNAME_PATTERN } from "./types";

export { PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH, USERNAME_PATTERN };

const USERNAME_RE = new RegExp(USERNAME_PATTERN);

/** Human version of `USERNAME_PATTERN` (form hint / validation message). */
export const USERNAME_HINT = "3–32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter or digit.";

/** Strip + lowercase, exactly what the server stores (`" Alice "` → `"alice"`). */
export function normalizeUsername(raw: string): string {
  return raw.trim().toLowerCase();
}

/** Validation message for a new username (after normalizing), or null if valid. */
export function usernameProblem(raw: string): string | null {
  const name = normalizeUsername(raw);
  if (name === "") return "Enter a username.";
  return USERNAME_RE.test(name) ? null : `Usernames are ${USERNAME_HINT}`;
}

/** Password length the way the server counts it: code points after NFKC normalization. */
export function passwordLength(password: string): number {
  return [...password.normalize("NFKC")].length;
}

/**
 * The server's password policy (same messages as `check_password_policy`): length 12..256 after
 * NFKC, not the username (case-insensitive), and different from `currentPassword` if given.
 * Returns the violation message or null.
 */
export function passwordPolicyViolation(password: string, username: string, currentPassword?: string): string | null {
  const pw = password.normalize("NFKC");
  const length = [...pw].length;
  if (length < PASSWORD_MIN_LENGTH) return `Use at least ${PASSWORD_MIN_LENGTH} characters.`;
  if (length > PASSWORD_MAX_LENGTH) return `Use at most ${PASSWORD_MAX_LENGTH} characters.`;
  if (pw.toLowerCase() === username.toLowerCase()) return "The password must not be the same as the username.";
  if (currentPassword !== undefined && pw === currentPassword.normalize("NFKC")) return "The new password must be different from the current one.";
  return null;
}

export interface NewPasswordInput {
  newPassword: string;
  confirmPassword: string;
  /** The account's username (`me.username`). */
  username: string;
  currentPassword?: string;
}

/** Field-level problems for the change-password form; empty object = OK to submit. */
export function newPasswordProblems(input: NewPasswordInput): { newPassword?: string; confirmPassword?: string } {
  const out: { newPassword?: string; confirmPassword?: string } = {};
  const violation = passwordPolicyViolation(input.newPassword, input.username, input.currentPassword || undefined);
  if (violation) out.newPassword = violation;
  if (input.confirmPassword !== input.newPassword) out.confirmPassword = "The passwords don't match.";
  return out;
}
