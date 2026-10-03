# Mimic — User management + authentication plan

> Adds admin-managed accounts, server-side sessions and per-owner job visibility to the MVP.
> Status: implemented (P0–P3 done, see the notes at the end). Branch `master`. Date: 2026-10-03. Companion to `docs/PLAN.md`
> (its §0 D1–D5 and P1–P9 still apply; nothing here reopens them).

---

## 0. Decisions

### 0.1 User decisions (final, do not reopen)

| # | Decision |
|---|---|
| U1 | Roles `admin` and `user`. No self-registration. Admins create users. |
| U2 | Admin can disable/enable (sessions revoked immediately), reset password (temporary password, forced change at next login), change role, hard-delete a user incl. their jobs + files. The **last active admin** cannot be demoted, disabled or deleted. |
| U3 | Every job has an owner. Users see only their own jobs, admins see all. Applies to status, result, video, keyframes, re-run interpretation and the new job list. |
| U4 | Every user may opt in to AI labeling per upload (default OFF). `POST /api/interpreter/check` is admin-only. `/api/health` stays but leaks nothing sensitive to unauthenticated callers. |

### 0.2 Decisions made in this plan

| # | Decision | Rationale |
|---|---|---|
| A1 | **Server-side sessions in SQLite.** Opaque token `secrets.token_urlsafe(32)` in a cookie; only `sha256(token)` is stored. Idle timeout 12 h, absolute 7 days (settings). No JWT. | Revocation (disable, role change, password change, logout) is a `DELETE`, which is exactly what U2 needs. One SQLite read per request is negligible next to the existing polling. |
| A2 | **Cookie** `mimic_session`: `HttpOnly; SameSite=Lax; Path=/; Max-Age=<absolute remaining>`, no `Domain` (host-only), `Secure` when `MIMIC_COOKIE_SECURE=1` (default 0 because dev runs on plain http). | `localhost:3000 → localhost:8000` is **same-site** (site = scheme + host; port is ignored), so a Lax cookie is sent on `fetch(..., {credentials:"include"})`, on the XHR upload with `withCredentials`, and on `<video>`/`<img>` subresource requests (no `crossorigin` attribute in the code today, so their credentials mode is "include"). |
| A3 | **Passwords: stdlib `hashlib.scrypt`.** N = 2^15, r = 8, p = 1, dklen 32, 16-byte random salt per hash, **`maxmem=64 MiB` passed explicitly** (N·r·128 = 32 MiB, which is at or over OpenSSL's default 32 MiB cap and raises `ValueError` without it). Stored as `scrypt$<log2N>$<r>$<p>$<salt b64>$<hash b64>`. Compare with `hmac.compare_digest`. A successful login re-hashes when the stored params differ from the current ones. At most 4 concurrent hashes (`threading.BoundedSemaphore`). | No new dependency. The encoded params allow raising the cost later. The semaphore caps memory at about 128 MiB under a login burst (FastAPI's threadpool has 40 threads). |
| A4 | **Password policy:** 12–256 characters after NFKC normalization, not equal to the username (case-insensitive), new ≠ current. No composition rules. | NIST 800-63B style: length over complexity. 12 characters is fine for a local tool. |
| A5 | **Temporary passwords are always server-generated** (`secrets.token_urlsafe(12)`, 16 chars) for both *create user* and *reset password*, shown **once** in the response, and set `must_change_password=1`. Admins never choose another person's long-term password. | One code path. The admin never knows the user's real password. |
| A6 | **Login throttling, in memory:** a sliding 15-minute window per `(client IP, normalized username)`: 5 failures → 429 `too_many_attempts` + `Retry-After`. Also per IP: 30 failures / 15 min. A success clears the `(ip, username)` key. The same limiter guards `POST /api/auth/password` (keyed by user id). Unknown usernames still run one dummy scrypt verify (constant time against enumeration). | A local single-process tool doesn't need persistence across restarts. Keying on IP + username avoids the "lock the admin out from anywhere" DoS that a username-only lockout creates. `X-Forwarded-For` is **not** trusted (no proxy in this setup). |
| A7 | **CSRF: Origin check on unsafe methods, no CSRF token.** A pure-ASGI middleware rejects `POST/PUT/PATCH/DELETE` whose `Origin` header is present and is neither in `MIMIC_CORS_ORIGINS` nor the API's own origin (`scheme://Host`, for Swagger `/docs`) → 403 `origin_not_allowed`. A request with no `Origin` passes, because browsers send `Origin` on every cross-origin request and every non-GET fetch/XHR/form POST, so a missing `Origin` means a non-browser client (curl, tests). | SameSite=Lax alone is **not enough here**: every other app on `localhost:*` is *same-site*, so a malicious or compromised local dev page could POST a multipart upload or `logout` with the cookie attached (multipart and empty-body POSTs are CORS "simple requests" with no preflight). The Origin check closes that hole with no client changes. A token adds client plumbing and protects nothing extra once Origin is enforced. |
| A8 | **CORS:** `allow_credentials=True`, `allow_origins = MIMIC_CORS_ORIGINS` (explicit list), methods `GET, POST, PATCH, DELETE, OPTIONS`. A settings validator **rejects `*`** in `MIMIC_CORS_ORIGINS` and fails startup. | With credentials and `*`, Starlette echoes any request origin, which would hand every site a credentialed CORS read. |
| A9 | **Route protection in Next 16: client-side gate, no `proxy.ts`.** A client `AuthGate` in the root layout resolves the session (`GET /api/auth/me`), shows a neutral loading state until it resolves (no protected content is rendered first), and redirects according to a pure `routeDecision()` from `web/lib`. Server-side authorization stays entirely in the API. | Checked against the bundled docs: `middleware` is now `proxy.ts` in Next 16 (`01-getting-started/16-proxy.md`). The docs say Proxy is for *optimistic* checks only and "should not be used as a full session management or authorization solution". Our pages are empty shells: all data is fetched from the browser to the API, and the API enforces authz. A proxy check would also only work while web and API share a hostname (the cookie is set by the API host), and mock mode has no real cookie. It would cost the most and protect the least. Revisit only if pages ever render data server-side. |
| A10 | **Legacy jobs (no owner):** the migration leaves `owner_id = NULL`. `NULL` means **admin-only** in every authz check. **The first admin created via the CLI claims all `NULL` jobs** (the CLI prints the count). | No admin exists at migration time, so jobs can't be assigned then. After the bootstrap no orphans remain. The `NULL` = admin-only rule is a safe default if one ever reappears. |
| A11 | **No admin yet:** `POST /api/auth/login` → 409 `setup_required` with a message naming `make create-admin`. The public `GET /api/auth/status` → `{setup_required}` lets the login page show setup instructions instead of a form. No default credentials are created. | The API stays useless, not open, until an operator runs the CLI on the host. |
| A12 | **`/api/health` stays public, same shape, nullable details.** Anonymous callers get `status`, `version`, `ffmpeg`, plus `interpreter: null` and `limits: null`. A full session (not in forced-password-change state) gets everything as today. | The README's `curl /api/health` → `"ffmpeg": true` setup check keeps working. The interpreter mode/model (infrastructure details) and limits are only for signed-in users. `version` and `ffmpeg` are not sensitive (open-source repo). |
| A13 | **Admin self-protection:** an admin cannot disable, delete or reset-password **their own** account (409 `self_action_forbidden`). Changing their own role is allowed unless they are the last active admin. | Prevents a one-click self-lockout. Self password changes go through `/api/auth/password`. |
| A14 | **Deleting a user with a running job:** rows and dirs are deleted immediately. The runner gets a **deleted-job guard**: after a run (any outcome) it re-reads the row, and if the row is gone it removes the job dir again (the pipeline may have re-created it) and skips all status writes. | No 409 friction for the admin and no orphan dirs left behind. |
| A15 | **Schema versioning via `PRAGMA user_version`** with ordered, idempotent migrations in one module (`core/db.py`), run at API startup and by the CLI, each inside `BEGIN IMMEDIATE`. Before migrating a v0 DB that has job rows, a `sqlite3` online backup is written to `data/mimic.db.pre-auth-<UTC>.bak`. If the DB version is newer than the code → refuse to start with a clear error. | Safe on an existing `data/mimic.db` (WAL, possibly with a running API). DDL is transactional in SQLite, so a failed migration leaves v0 intact. |
| A16 | **`make clean-jobs` keeps accounts.** It now runs `app.cli clean-jobs --yes` (deletes job rows + `data/jobs/*`, keeps users and sessions). The old full wipe moves to `make reset-data` (DB + jobs, needs `create-admin` again). | Today it `rm`s `mimic.db`, which would silently delete every account. |
| A17 | **Usernames:** `^[a-z0-9][a-z0-9._-]{2,31}$`, stored lowercase, unique. No email, no display name. | Org policy: minimize PII. Nothing in the product needs more. |
| A18 | **`JobStatus` gains `owner: {id, username} \| null`**. The job list reuses `JobStatus` items (`JobList {items, next_cursor}`). | One model, so the list, the job page and the mock share the same shape. |

### 0.3 Rejected

- **JWT / stateless tokens:** immediate revocation (U2) would need a denylist, which is a session table anyway.
- **Next `proxy.ts` / server-side session checks:** see A9.
- **Proxying the API through Next rewrites to get a first-party cookie:** reverses PLAN P7 (upload size limits, XHR progress).
- **CSRF double-submit token:** see A7.
- **Username-only lockout:** see A6.
- **Admin-chosen initial passwords:** see A5.
- **Assigning legacy jobs at migration time:** no admin exists yet (A10).
- **New dependencies (passlib, argon2-cffi, itsdangerous, an auth library):** stdlib covers it.

---

## 1. Context (codebase findings that shape this plan)

- `api/app/main.py` builds `Services(settings, storage, store, runner)` in the lifespan. CORS today: no credentials, methods `GET/POST/OPTIONS`.
- `api/app/api/deps.py::get_job` is the **single choke point** for every job-by-id route (`JobDep`), which makes it the natural place for owner checks. `routes_jobs.py` routes: create, status, result, interpret, video, keyframes.
- `SqliteJobStore` opens one short-lived `sqlite3` connection per operation, uses WAL, and creates its schema in `init()` via `executescript` (which implicitly COMMITs, so it can't be used inside a migration transaction).
- `schemas.py` is the frozen contract. `make contract` generates `web/lib/types.ts` + `docs/contract/motion-spec.schema.json` and `test_ir_contract.py` fails if they are stale. In serialization mode every field is *required* (nullable where it applies), so adding a field to `JobStatus`/`Health` breaks web typecheck until the consumers are updated (mock, `api.ts` validators, 4 UI files that read `health.interpreter`).
- `web/lib/errors.ts` (`Record<ErrorCode, true>`) and `web/lib/format.ts` (`Record<ApiErrorCode, ErrorDescription>`) **fail typecheck when a backend ErrorCode is added** without listing it. Both must change in the same step as `errors.py`.
- `web/lib/api.ts`: `requestJson` uses `fetch` without `credentials`. `uploadVideo` uses XHR without `withCredentials`. `looksLikeHealth` requires `interpreter` to be an object.
- `api/tests/unit/test_api.py` builds clients through `make_client` (plus one direct `TestClient`). It exercises every job route unauthenticated, so it must move to authenticated clients. `TestClient` sends no `Origin`, so A7 doesn't affect it.
- `scripts/analyze.py`, `scripts/eval_synth.py` and the synth tests call the pipeline directly and never touch the DB or HTTP. They stay auth-free if nothing in `app.pipeline` imports `app.auth` (enforced by a test).
- `Makefile clean-jobs` deletes `mimic.db*` (see A16).
- Vault: `recall.py --brief` could not be run (no shell tool in this session) and vault writes were excluded by the orchestrator. Project decisions are taken from `docs/PLAN.md` (its §0 and Phase 4/5 notes).

---

## 2. Data model (`core/db.py`, schema v1)

```sql
-- existing, unchanged columns
CREATE TABLE IF NOT EXISTS jobs ( id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0, error_code TEXT, error_message TEXT, original_filename TEXT NOT NULL,
  ext TEXT NOT NULL, options_json TEXT NOT NULL, source_json TEXT, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL );
ALTER TABLE jobs ADD COLUMN owner_id TEXT;            -- only if PRAGMA table_info lacks it; NULL = legacy/admin-only
CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs (owner_id, created_at, id);
CREATE INDEX IF NOT EXISTS jobs_created       ON jobs (created_at, id);
-- (existing jobs_status_created stays)

CREATE TABLE IF NOT EXISTS users (
  id                   TEXT PRIMARY KEY,                        -- uuid4().hex
  username             TEXT NOT NULL UNIQUE CHECK (username = lower(username)),
  password_hash        TEXT NOT NULL,                           -- scrypt$15$8$1$<salt>$<hash>
  role                 TEXT NOT NULL CHECK (role IN ('admin','user')),
  is_active            INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
  must_change_password INTEGER NOT NULL DEFAULT 0 CHECK (must_change_password IN (0,1)),
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL,
  password_changed_at  TEXT NOT NULL,
  last_login_at        TEXT,
  created_by           TEXT                                     -- informational user id, no FK
);
CREATE INDEX IF NOT EXISTS users_role_active ON users (role, is_active);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash   TEXT PRIMARY KEY,                                -- sha256(token) hex
  user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at   TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  expires_at   TEXT NOT NULL                                    -- absolute expiry
);
CREATE INDEX IF NOT EXISTS sessions_user    ON sessions (user_id);
CREATE INDEX IF NOT EXISTS sessions_expires ON sessions (expires_at);
```

- Times are ISO-8601 UTC strings, as in the existing `jobs` table.
- `jobs.owner_id` gets **no FK** (`ALTER TABLE ADD COLUMN` with a FK is allowed but buys nothing). User deletion removes jobs explicitly because their files must go too.
- No IP or user-agent columns (PII minimization).
- `db.connect(path)` (shared by all stores): `sqlite3.connect(timeout=10)`, `row_factory=Row`, `PRAGMA foreign_keys=ON` on every connection (needed for the sessions cascade), WAL set once in `migrate`. Manual transactions use `isolation_level=None` + explicit `BEGIN IMMEDIATE`.
- `last_seen_at` is written only when it is ≥ 60 s old, so the 800 ms job polling doesn't turn every read into a write.
- Expired sessions are purged opportunistically on every successful login (`DELETE ... WHERE expires_at < now OR last_seen_at < now - idle`).

### 2.1 Migration (`core/db.py::migrate(db_path)`)

1. `mkdir` the parent dir and open a connection. Read `PRAGMA user_version` (v). If v > `SCHEMA_VERSION` (1) → raise `RuntimeError("database schema vN is newer than this code (v1); upgrade mimic")`.
2. If v == 0, the `jobs` table exists and `SELECT count(*) FROM jobs` > 0 → online backup (`src.backup(dst)`) to `mimic.db.pre-auth-<YYYYmmddTHHMMSSZ>.bak` in the data dir, and log the path.
3. `PRAGMA journal_mode=WAL` (outside the transaction).
4. `BEGIN IMMEDIATE`, re-read `user_version` (another process may have migrated meanwhile), then apply each pending step with individual `execute` calls (never `executescript`). Steps are idempotent (`IF NOT EXISTS`, `table_info` check). `PRAGMA user_version = 1`, `COMMIT`. On any exception → `ROLLBACK`, re-raise.
5. Fresh DB → the same path (creates `jobs` + the rest), no backup.

`SqliteJobStore.init()` calls `db.migrate()` (keeps `test_jobstore.py` and the lifespan working). The CLI calls it too. Downgrade: unsupported. Old code would still run against a v1 DB (its `INSERT`s list columns explicitly and it ignores the new tables), but its jobs would be ownerless. Documented, not tested.

---

## 3. API contract

All errors keep `{"error": {"code", "message"}}`. **Every response under `/api/auth/*` and `/api/admin/*` carries `Cache-Control: no-store`.** Request bodies: `extra="forbid"` (existing `ApiModel`).

### 3.1 New / changed models (`api/app/api/schemas.py`, exported via `CONTRACT_MODELS`)

| Model | Fields |
|---|---|
| `UserRole` (literal, named in `contract.NAMED_LITERALS`) | `"admin" \| "user"` |
| `JobOwner` | `id: str`, `username: str` |
| `Me` | `id`, `username`, `role: UserRole`, `must_change_password: bool`, `created_at` |
| `AuthStatus` | `setup_required: bool` |
| `LoginRequest` | `username: str` (1..64, stripped + lowercased server-side), `password: str` (1..1024) |
| `ChangePasswordRequest` | `current_password: str` (1..1024), `new_password: str` (1..1024; policy checked in code so the error is `weak_password`, not `invalid_request`) |
| `AdminUser` | `id`, `username`, `role`, `is_active: bool`, `must_change_password: bool`, `created_at`, `updated_at`, `last_login_at: datetime \| null`, `job_count: int` |
| `AdminUserList` | `items: AdminUser[]` (sorted by username; no pagination) |
| `CreateUserRequest` | `username: str` (pattern A17), `role: UserRole = "user"` |
| `UpdateUserRequest` | `role: UserRole \| None = None`, `is_active: bool \| None = None`; validator: at least one set |
| `TemporaryPassword` | `user: AdminUser`, `temporary_password: str` |
| `JobList` | `items: JobStatus[]`, `next_cursor: str \| null` |
| `JobStatus` (changed) | `+ owner: JobOwner \| None = None` (null only for legacy orphan jobs) |
| `Health` (changed) | `interpreter: HealthInterpreter \| None`, `limits: HealthLimits \| None` (null for anonymous callers) |

`contract._ts_constants` also exports `PASSWORD_MIN_LENGTH = 12`, `PASSWORD_MAX_LENGTH = 256` and `USERNAME_PATTERN` (string), defined in `app/auth/policy.py` (P0) as the single source of truth.

### 3.2 New error codes (`core/errors.py`)

| Code | HTTP | Default message (tone of existing ones) |
|---|---|---|
| `unauthenticated` | 401 | "You are not signed in, or your session expired. Sign in again." |
| `invalid_credentials` | 401 | "Wrong username or password." |
| `account_disabled` | 403 | "This account is disabled. Ask an admin to enable it." |
| `forbidden` | 403 | "You don't have permission to do this." |
| `password_change_required` | 403 | "Set a new password before continuing." |
| `origin_not_allowed` | 403 | "This request came from a page that is not allowed to use this API." |
| `too_many_attempts` | 429 | "Too many failed attempts. Wait a few minutes and try again." (+ `Retry-After` seconds) |
| `setup_required` | 409 | "No admin account exists yet. On the API host, run `make create-admin`." |
| `last_admin` | 409 | "This is the last active admin. Make another user an admin first." |
| `self_action_forbidden` | 409 | "You can't do this to your own account." |
| `username_taken` | 409 | "That username is already taken." |
| `weak_password` | 422 | "Passwords need at least 12 characters and must not match the username." (specific text per violation) |
| `current_password_incorrect` | 422 | "The current password is wrong." (422 on purpose: a 401 would read as "session expired" to the client) |

`main._HTTP_STATUS_CODES` gains `401 → unauthenticated`, `403 → forbidden`, `429 → too_many_attempts` for errors raised by Starlette itself.

### 3.3 Endpoints

**Auth (Track B1):**

| Method & path | Auth | Request | Success | Errors |
|---|---|---|---|---|
| `GET /api/auth/status` | public | — | 200 `AuthStatus` | — |
| `POST /api/auth/login` | public | `LoginRequest` | 200 `Me` + `Set-Cookie` (new session; any presented old cookie is revoked) | 401 `invalid_credentials`, 403 `account_disabled` (only after a correct password), 409 `setup_required`, 422 `invalid_request`, 429 `too_many_attempts`, 403 `origin_not_allowed` |
| `POST /api/auth/logout` | any | — | 204, deletes the session row, `Set-Cookie` expiring the cookie. Idempotent (204 without a session) | 403 `origin_not_allowed` |
| `GET /api/auth/me` | session (forced change allowed) | — | 200 `Me` | 401 `unauthenticated` |
| `POST /api/auth/password` | session (forced change allowed) | `ChangePasswordRequest` | 200 `Me` (`must_change_password:false`) + `Set-Cookie` with a **new** session. All of the user's sessions are revoked first | 401, 422 `current_password_incorrect` / `weak_password`, 429 |

**Admin (Track B1). All need an admin with a full session.** 401 if anonymous, 403 `forbidden` for users, 403 `password_change_required` while forced. Unknown or malformed `user_id` (`^[0-9a-f]{32}$`) → 404 `not_found`.

| Method & path | Request | Success | Errors / effects |
|---|---|---|---|
| `GET /api/admin/users` | — | 200 `AdminUserList` | — |
| `POST /api/admin/users` | `CreateUserRequest` | 201 `TemporaryPassword` | 409 `username_taken`, 422 |
| `PATCH /api/admin/users/{id}` | `UpdateUserRequest` | 200 `AdminUser` | 409 `last_admin` (demote/disable the last active admin), 409 `self_action_forbidden` (`is_active:false` on self). **Role change or disable → all of the target's sessions are revoked.** Enable does not create sessions. No-op values return 200 unchanged |
| `POST /api/admin/users/{id}/reset-password` | — | 200 `TemporaryPassword` | 409 `self_action_forbidden`. Sets `must_change_password=1` and revokes the target's sessions |
| `DELETE /api/admin/users/{id}` | — | 204 | 409 `last_admin`, 409 `self_action_forbidden`. One transaction: delete the target's sessions, jobs rows and user row. After commit: `storage.delete_job` for each removed job id (best effort, failures logged) |

The last-admin guard is atomic: inside `BEGIN IMMEDIATE`, count active admins excluding the target, then update or delete. Two admins demoting each other concurrently can't both succeed.

**Jobs (Track B2).** Every route requires a full session.

| Method & path | Change |
|---|---|
| `GET /api/jobs` **(new)** | Query: `owner` (`me` or a user id; admin only for ids other than self → otherwise 403 `forbidden`), `status` (`JobState`, optional), `limit` (1..200, default 50), `cursor` (opaque). Order `created_at DESC, id DESC`, keyset pagination; `cursor` = urlsafe-b64 of `"<created_at>\|<id>"` (bad cursor → 422). Non-admin: always only own jobs. Admin without `owner`: all jobs incl. orphans. → 200 `JobList` |
| `POST /api/jobs` | Records `owner_id = current user`. Everything else unchanged |
| `GET /api/jobs/{id}`, `/result`, `/video`, `/keyframes/{name}`, `POST /api/jobs/{id}/interpret` | `get_job` checks visibility: admin → any. User → `owner_id == user.id`. Otherwise **404 `not_found`** (never 403, so existence doesn't leak). `JobStatus.owner` is filled |
| `GET /api/health` | Optional session. Anonymous (or forced change) → `interpreter:null, limits:null` |
| `POST /api/interpreter/check` | Admin only (403 `forbidden` for users) |

Check order on every protected route: Origin (middleware) → session (401) → forced change (403) → role (403) → resource (404).

---

## 4. Authorization rules

| Action | Anonymous | User (forced change) | User | Admin |
|---|---|---|---|---|
| `GET /api/health` | public subset | public subset | full | full |
| `GET /api/auth/status`, `POST /api/auth/login` | yes | yes | yes | yes |
| `GET /api/auth/me`, `POST /api/auth/password`, `POST /api/auth/logout` | 401 (logout: 204) | yes | yes | yes |
| `POST /api/jobs` (incl. `use_interpreter=true`) | 401 | 403 | yes, owner = self | yes, owner = self |
| `GET /api/jobs` | 401 | 403 | own only | all (+ `owner` filter) |
| Job by id: status / result / video / keyframes / interpret | 401 | 403 | own: yes · other's or orphan: 404 | any |
| `POST /api/interpreter/check` | 401 | 403 | 403 `forbidden` | yes |
| `/api/admin/users*` | 401 | 403 | 403 `forbidden` | yes (guards: `last_admin`, `self_action_forbidden`) |
| Disabled user (any route) | — | — | sessions were deleted → 401; login → 403 `account_disabled` | same |
| Expired session (idle or absolute) | — | 401 + row deleted | 401 + row deleted | 401 + row deleted |

Resolving a session is one query: `sessions JOIN users` on `token_hash`. It requires `users.is_active = 1` (defence in depth on top of the revocation), `expires_at > now` and `last_seen_at > now − idle`. Role and `must_change_password` come from the **users row at request time**, never from the session.

---

## 5. Sessions, cookies, CSRF, rate limiting: implementation notes

- **Settings (`config.py`)**: `session_idle_minutes: int = 720` (>0), `session_absolute_hours: int = 168` (>0), `cookie_secure: bool = False`; `cors_origins` validator rejects `"*"`. The cookie name `mimic_session` is a constant, not a setting.
- **Setting the cookie**: `response.set_cookie("mimic_session", token, max_age=<absolute seconds>, httponly=True, samesite="lax", secure=settings.cookie_secure, path="/")`. Clearing it uses `delete_cookie` with the same `path`/`samesite`/`secure`.
- **Never log** passwords, tokens, token hashes or temporary passwords. Audit lines log only ids: `auth: user <id> logged in`, `admin <id> disabled user <id>`, `admin <id> deleted user <id> (<n> jobs)`. Failed logins log a count, not the username.
- **Hashing runs off the event loop**: auth routes are plain `def` (FastAPI's threadpool) or wrap hashing in `run_in_threadpool`.
- **Origin middleware (`app/auth/csrf.py`)**: pure ASGI (not `BaseHTTPMiddleware`, so the streaming upload body is never wrapped). It reads only headers. Registered in `main.py` *after* `CORSMiddleware`, so preflights stay CORS's job and a rejection still gets CORS headers for a readable error. Allowed set = `settings.cors_origins` ∪ `{f"{scheme}://{host header}"}`. Compare normalized lowercase `scheme://host[:port]`; `Origin: null` is rejected.
- **Hostname caveat (documented):** cookies are host-scoped and ignore ports. That makes `localhost:3000 → localhost:8000` work, but a mix of `127.0.0.1` and `localhost` is *cross-site*: the browser drops the cookie and login appears to "not stick". The web client detects this (W-lib `cookie_rejected`). The same host-scoping means **any other server on `localhost:*` receives the cookie** on requests the browser makes to it. That is acceptable for a local tool (HttpOnly, expiring), and it is listed in the risks.

---

## 6. CLI bootstrap (`api/app/cli.py`, `python -m app.cli`)

| Command | Behavior |
|---|---|
| `create-admin [--username NAME]` | Runs `db.migrate`. Prompts for the username if missing (validated by A17), then the password twice via `getpass` (policy A4; mismatch → re-prompt, max 3 tries). With `--password-stdin` it reads one line from stdin instead (for scripts/tests; never argv). Creates an active admin with `must_change_password=0`. **If it is the first admin**: `UPDATE jobs SET owner_id=? WHERE owner_id IS NULL` and print "Assigned N existing jobs to <name>". If an active admin already exists: still allowed (break-glass), prints a note. Exit codes: 0 ok, 2 invalid input, 3 username taken |
| `reset-password USERNAME` | Break-glass for operators: prompts for a new password (or `--password-stdin`), sets it with `must_change_password=0`, **enables the account**, revokes all of its sessions. Works for any role |
| `list-users` | Table: username, role, active, must-change, jobs. No hashes |
| `clean-jobs --yes` | Deletes all job rows and `data/jobs/*`, keeps users and sessions (A16). Without `--yes` → prints what it would do and exits 1 |

Makefile: `create-admin` (`cd api && uv run python -m app.cli create-admin`), `reset-password` (`USER=<name>`), `clean-jobs` → CLI, new `reset-data` (old `rm -rf` behavior + note to re-run `create-admin`). `help` text updated.

---

## 7. Backend module layout

```
api/app/
  auth/
    __init__.py
    models.py      [P0] Role(StrEnum), UserRecord, SessionRecord (frozen dataclasses), AuthServices dataclass
                        (users, sessions, limiter, settings) — the object stored on app.state.auth
    policy.py      [P0] USERNAME_RE / USERNAME_PATTERN, PASSWORD_MIN/MAX_LENGTH, normalize_username(),
                        check_password_policy(pw, username) -> str | None (violation message)
    deps.py        [P0 signatures → B1 bodies] get_auth(request) -> AuthServices;
                        OptionalUserDep (UserRecord | None, full sessions only, never raises);
                        SessionUserDep (allows forced change; 401 otherwise);
                        CurrentUserDep (full session; 401/403 password_change_required);
                        AdminDep (CurrentUserDep + role admin; 403 forbidden).
                        P0 bodies raise PipelineError(UNAUTHENTICATED) / return None.
    passwords.py   [B1] hash_password, verify_password, needs_rehash, generate_temporary_password, dummy verify
    sessions.py    [B1] SqliteSessionStore: create(user_id)->token, resolve(token)->(SessionRecord, UserRecord)|None,
                        revoke(token), revoke_all(user_id), purge_expired(); cookie set/clear helpers
    users.py       [B1] SqliteUserStore: create, get, get_by_username, list_with_job_counts, update(role/is_active,
                        atomic last-admin guard), set_password(must_change), record_login, delete(atomic; returns
                        removed job ids), has_active_admin, claim_orphan_jobs(user_id)
    ratelimit.py   [B1] LoginLimiter (thread-safe sliding window, injectable clock)
    csrf.py        [B1] OriginCheckMiddleware (pure ASGI)
  core/db.py       [P0] connect(), migrate(), SCHEMA_VERSION
  api/routes_auth.py   [P0 empty router → B1]
  api/routes_admin.py  [P0 empty router → B1]
  cli.py           [B1]
```

`get_auth` reads `request.app.state.auth`, which `main.py`'s lifespan builds (B1). `app.state.services` / `Services` (B2's `api/deps.py`) stay unchanged. The two tracks never edit the same file.

---

## 8. Frontend

### 8.1 Routes and states (UI by ui-ux-master; this fixes pages, data, states only)

| Route | Access | Data | States |
|---|---|---|---|
| `/login` (server page: `await props.searchParams` → passes a validated `next` to a client form, so no `useSearchParams`/Suspense is needed) | public; an authenticated user is redirected to `next` or `/` | `getAuthStatus()`, `login()` | loading status; **setup required** (no form, shows `make create-admin` instructions and a "Check again" button); idle; submitting; `invalid_credentials`; `account_disabled`; `too_many_attempts` (countdown from `retryAfterS`); network error; **`cookie_rejected`** (login OK but `/me` 401 → "use the same hostname for web and API, e.g. both `localhost`") |
| `/account/password` | session (forced allowed) | `changePassword()` | forced variant (explanation, no app nav except sign out); voluntary variant; client checks (length ≥ `PASSWORD_MIN_LENGTH`, confirm matches); `current_password_incorrect`; `weak_password` (server message); submitting; success → `next` or `/` |
| `/` (upload) | full session | unchanged | **Test connection button only for admins**. Users see interpreter status without the button. AI opt-in toggle unchanged for everyone |
| `/jobs` **(new)** "My jobs" / "All jobs" | full session | `useJobList({ owner, status })` | loading; empty (CTA to upload); list: filename, created, status/stage + progress for running jobs, duration, AI on/off, link to `/jobs/<id>`; **admins: Owner column ("Legacy" for `owner:null`) + owner filter** (All / Me / each user from `listUsers()`); "Load more" (cursor); error + retry; refreshes every 5 s while any row is non-terminal and on window focus |
| `/jobs/[id]` | full session | unchanged | `not_found` copy becomes "doesn't exist or you don't have access". Admins viewing someone else's job see "Owner: <username>" |
| `/admin/users` **(new)** | admin | `useAdminUsers()` | loading; list (username, role, active/disabled, "must change password", jobs, last login, created); **non-admin → 403 state** (not a redirect); per row: change role, disable/enable, reset password, delete. **Own row**: no disable/delete/reset. **Last active admin**: demote/disable/delete disabled with a reason, and server 409 `last_admin` handled anyway. Create dialog (username + role) → **temporary password reveal** (shown once, copy button, "won't be shown again"). Reset → the same reveal. Delete confirmation states the job count that will be deleted and requires typing the username. Role change on self → confirm "you will be signed out" (sessions revoked) |
| Header (`SiteHeader`) | — | `useSession()` | anonymous (logo only); authenticated: nav Upload · Jobs · Users (admin) + user menu (username, role badge for admin, Change password, Sign out); Mock API badge stays |

`AuthGate` (client, root layout) uses `useSession()` + `usePathname()` + `useRouter()` and applies `routeDecision()`. While the session is `loading` it renders a neutral placeholder, never the page.

### 8.2 `web/lib` changes (W-lib)

- `api.ts`:
  - `requestJson`: `credentials: "include"`; methods `GET|POST|PATCH|DELETE`; 204 → `undefined`.
  - `uploadVideo`: `xhr.withCredentials = true`.
  - `looksLikeHealth`: no longer requires `interpreter`.
  - New: `getAuthStatus()`, `login(username, password)`, `logout()`, `getMe()`, `changePassword(current, next)`, `listJobs({ owner?, status?, limit?, cursor?, signal? }) → JobList`, `listUsers()`, `createUser({ username, role })`, `updateUser(id, { role?, isActive? })`, `resetUserPassword(id)`, `deleteUser(id)`.
  - Every HTTP path (fetch + XHR): a 401 `unauthenticated` → `notifyAuthEvent("unauthenticated")`, a 403 `password_change_required` → `notifyAuthEvent("password_change_required")` (tiny listener registry `onAuthEvent(cb)`).
  - `ApiError` gets `retryAfterS: number | null` (from `Retry-After`).
- `errors.ts`: new backend codes. New client code `cookie_rejected`. `isTransient` stays false for every new code.
- `format.ts`: `describeError` entries for the new codes (`setup_required` guidance = the CLI command), `ROLE_LABELS`, `formatLastLogin`.
- `session.ts` **(new, no JSX)**: module store with `useSyncExternalStore`. State `{ status: "loading" | "anonymous" | "authenticated" | "error", me: Me | null, error }`. Functions `loadSession()`, `login()` (login → `getMe()`; 401 here → `cookie_rejected`), `logout()`, `changePassword()`, `refreshSession()`. It subscribes to `onAuthEvent` (unauthenticated → anonymous; password_change_required → refresh). Hook `useSession()`. No React provider needed.
- `routes.ts` **(new, pure)**: `routeDecision(pathname, session) → { kind: "loading" } | { kind: "render" } | { kind: "forbidden" } | { kind: "redirect", to }` implementing §8.1. `safeNext(raw)` accepts only same-app paths (`/` prefix, not `//`, no scheme or backslash) and falls back to `/`.
- `useJobList.ts`, `useAdminUsers.ts` **(new hooks)**: loading/error/refresh, pagination (`loadMore`), mutations returning `ApiError` for the UI (`last_admin`, `self_action_forbidden`, `username_taken`).
- `useJobPolling` / `jobPoller` unchanged: 401/403/404 are already final. The session store handles redirects.
- `README.md`: "Auth" section (session store, route policy, error codes, mock users).

### 8.3 Mock mode (`NEXT_PUBLIC_API_MOCK=1`)

- New `mock/mockAuth.ts`: in-memory users + current session, persisted to `localStorage` (`mimic.mock.auth`). Node tests use an in-memory fallback. Seeded users: `admin` (admin), `user` (user), `newbie` (user, `must_change_password`), `disabled` (user, inactive). **Any non-empty password works except the literal `wrong`** → `invalid_credentials`. This is a fake for UI work, not security, and the README and the login page's mock badge say so. Login-page query `?fail=too_many_attempts|setup_required|cookie_rejected` simulates those states. Admin operations (create/update/reset/delete) mirror the real guards (`last_admin`, `self_action_forbidden`, `username_taken`). Temporary passwords are random strings.
- `mockApi.ts`: every `JobStatus` gets `owner`. Created jobs are owned by the current mock user (owner kept in the in-memory/session map). The fixture job `MOCK_SAMPLE_JOB_ID` is owned by `user`. Unknown-but-well-formed ids after a reload count as owned by the current user (lenient, same as today's id-encoded scenarios). Job routes apply U3 (other user's job → `not_found`). `mockListJobs` supports the owner filter. `mockCheckInterpreter` → `forbidden` for non-admins. `mockGetHealth` returns null details when anonymous.

---

## 9. Tests

**Backend (new):**

| File | Track | Covers |
|---|---|---|
| `tests/unit/test_db_migrate.py` | P0 | Fresh DB → v1 schema. **Existing v0 DB with rows** (built from the old DDL) → rows intact, `owner_id` NULL, indexes present, `user_version=1`, backup file written. Second run is a no-op. Newer-version DB → error. Failure mid-migration rolls back (monkeypatched step raises) |
| `tests/unit/test_jobstore.py` (extended) | P0 | owner on create/get, `list_jobs` keyset order + filters + cursor round-trip, `delete_by_owner`, orphan claim |
| `tests/auth_helpers.py` | P0 | `insert_user(db_path, username, role, *, active=True, must_change=False) -> id` (raw SQL, unusable hash), `as_user(app, user | None)` context manager using `app.dependency_overrides` for the deps in `app/auth/deps.py` |
| `tests/unit/test_passwords.py` | B1 | round trip, wrong password, malformed hash → False (no exception), unique salts, `needs_rehash`, explicit `maxmem` (real N=2^15 once), `compare_digest` used, policy messages, NFKC |
| `tests/unit/test_sessions.py` | B1 | create/resolve, hash-only storage, idle expiry, absolute expiry (injected clock), `last_seen` throttle, revoke / revoke_all, disabled user never resolves, cascade on user delete, purge |
| `tests/unit/test_users_store.py` | B1 | create/unique/normalize, last-admin guard on demote/disable/delete (incl. two admins demoting each other in a race via threads), delete returns job ids, job counts |
| `tests/unit/test_ratelimit.py` | B1 | window, per-key + per-IP limits, reset on success, `Retry-After` |
| `tests/unit/test_auth_api.py` | B1 | login/logout/me/password lifecycle, cookie attributes (HttpOnly, SameSite=Lax, Path, Max-Age; `Secure` with the setting), `setup_required`, `account_disabled`, 429, forced change gating, **Origin check** (foreign origin 403, allowed origin ok, missing origin ok, own origin ok, `null` 403), CORS preflight with credentials, `no-store` header, `*` origin rejected by settings |
| `tests/unit/test_admin_api.py` | B1 | create → temp password works once + forces change; reset revokes sessions; disable revokes immediately (next request 401); role change revokes; self guards; last-admin 409s; delete removes rows + dirs + sessions; non-admin 403 |
| `tests/unit/test_cli.py` | B1 | create-admin via `--password-stdin` / monkeypatched getpass, weak password rejected, first admin claims orphans, second admin doesn't, reset-password re-enables + revokes, clean-jobs keeps users, no password in argv parsing |
| `tests/unit/test_api.py` (migrated) | B2 | `make_client` gains `as_user=` (default: an admin via `as_user`). Existing assertions unchanged, `test_cors_preflight` expects credentials headers. New: owner recorded, `JobStatus.owner`, `/api/jobs` list + filters, health anonymous subset, interpreter check 403 for users |
| `tests/unit/test_runner_deleted_job.py` | B2 | job row deleted mid-run → dir removed at the end, no status writes, no exception |
| `tests/unit/test_authz_matrix.py` | INT | **Real login, no overrides.** Parametrized over every job route × {anonymous, owner, other user, admin, disabled, expired session, forced change} + orphan job + list visibility, expecting §4 exactly. Also asserts `app.pipeline` and `scripts/analyze.py` import nothing from `app.auth` |

- **Test speed:** an autouse fixture in `tests/conftest.py` (P0) monkeypatches the scrypt cost to N = 2^10 (hashes encode their params, so verification still works). Exactly one test uses the real cost.
- **Green bar:** the existing 665 pass (plus new tests); `make eval` SUITE PASS unchanged; `make contract-check`; `make lint`.

**Web (node:test, W-lib):** `routes.test.ts` (every row of §8.1 incl. forced change and admin-only), `safeNext` cases (`//evil`, `https://`, `/\\`, encoded), `session.test.ts` (against the mock: login → authenticated, `wrong` → error, `cookie_rejected` path, unauthenticated event → anonymous), `mock/mockAuth.test.ts` (guards mirror the backend), `api.mock-mode.test.ts` (job of another mock user → `not_found`, list filter, interpreter check forbidden for `user`). Existing tests stay green.

---

## 10. File ownership per track (no overlap within a phase)

| Track | Owner | Files |
|---|---|---|
| **P0** contract + schema + data layer | coder (1, sequential) | `api/app/api/schemas.py`, `api/app/core/errors.py`, `api/app/config.py`, `api/app/contract.py`, `docs/contract/motion-spec.schema.json` (gen), `web/lib/types.ts` (gen), `api/app/core/db.py` (new), `api/app/core/jobstore.py`, `api/app/auth/{__init__,models,policy,deps}.py` (new; deps = signatures + stub bodies), `api/app/api/routes_auth.py` + `routes_admin.py` (new, empty routers), `api/tests/conftest.py`, `api/tests/auth_helpers.py` (new), `api/tests/unit/test_db_migrate.py` (new), `api/tests/unit/test_jobstore.py`. **Keep-green web shims (mechanical only):** `web/lib/errors.ts` + `web/lib/format.ts` (new codes), `web/lib/api.ts` (`looksLikeHealth`), `web/lib/mock/mockApi.ts` (`owner: null`), null guards for `health.interpreter`/`limits` in `web/components/upload/{UploadOptions,UploadScreen,InterpreterStatus}.tsx`, `web/components/job/JobView.tsx`, `web/lib/upload.ts` if needed |
| **B1** backend auth core | coder | `api/app/auth/{passwords,sessions,users,ratelimit,csrf}.py` (new), `api/app/auth/deps.py` (bodies), `api/app/api/routes_auth.py`, `api/app/api/routes_admin.py`, `api/app/main.py` (lifespan builds `AuthServices` → `app.state.auth`; CORS credentials + methods; Origin middleware; include auth/admin routers; `_HTTP_STATUS_CODES`), `api/app/cli.py` (new), `Makefile`, tests `test_passwords/test_sessions/test_users_store/test_ratelimit/test_auth_api/test_admin_api/test_cli.py` |
| **B2** authz on existing routes | coder | `api/app/api/deps.py` (`get_job` takes `CurrentUserDep`, visibility check), `api/app/api/routes_jobs.py` (owner on create, `GET /api/jobs`, `to_job_status` owner), `api/app/api/routes_health.py` (`OptionalUserDep`), `api/app/api/routes_interpreter.py` (`AdminDep`), `api/app/core/runner.py` (deleted-job guard), `api/tests/unit/test_api.py`, `api/tests/unit/test_runner_deleted_job.py` (new) |
| **W-lib** | coder | `web/lib/api.ts`, `web/lib/errors.ts`, `web/lib/format.ts`, `web/lib/session.ts` (new), `web/lib/routes.ts` (new), `web/lib/useJobList.ts` (new), `web/lib/useAdminUsers.ts` (new), `web/lib/mock/mockAuth.ts` (new), `web/lib/mock/mockApi.ts`, `web/lib/README.md`, tests `web/lib/{routes,session}.test.ts`, `web/lib/mock/mockAuth.test.ts`, `web/lib/api.mock-mode.test.ts`, `web/lib/mock/mockApi.test.ts` |
| **W-ui** | ui-ux-master | `web/app/layout.tsx`, `web/app/login/page.tsx`, `web/app/account/password/page.tsx`, `web/app/jobs/page.tsx`, `web/app/admin/users/page.tsx` (new pages), `web/components/SiteHeader.tsx`, `web/components/auth/*` (AuthGate, LoginForm, SetupRequired, ChangePasswordForm, UserMenu, Forbidden), `web/components/jobs/*` (JobList, filters), `web/components/admin/*` (UsersTable, CreateUserDialog, TemporaryPasswordReveal, ConfirmDeleteDialog, RoleSelect), `web/components/upload/{UploadScreen,InterpreterStatus}.tsx` (admin-only Test connection), `web/components/job/{JobView,JobErrorState}.tsx` (owner line, 404 copy) |
| **INT** | coder | `api/tests/unit/test_authz_matrix.py` (new), `README.md` (Accounts section: create-admin, roles, hostname caveat, `MIMIC_COOKIE_SECURE`, session settings, clean-jobs vs reset-data), `.env.example` (new vars, no secrets), `web/.env.local.example` (comment on same hostname), `docs/PLAN.md` (short addendum pointing here), fixes found in E2E |

The P0 keep-green shims touch W-lib, W-ui and B2 files **before** those tracks start (sequential), and only mechanically (type additions, null guards).

---

## 11. Phases

```
P0 (sequential) ─┬─► B1 backend auth core ─┐
                 ├─► B2 authz existing      ├─► P2: W-ui ‖ INT-backend (authz matrix) ─► P3 INT E2E + docs
                 └─► W-lib (against mock) ──┘
```

### P0 — Contract, schema, data layer (M, 1 coder, blocks everything)
1. `errors.py` codes + statuses + messages. `config.py` settings + `*` validator. `auth/policy.py`, `auth/models.py`.
2. `schemas.py` models (§3.1) + `CONTRACT_MODELS`. `contract.py`: `UserRole` literal + constants. Run `make contract`.
3. `core/db.py` (connect, migrate v1, backup). `jobstore.py`: switch to `db.connect`, `init()` → `migrate`, `owner_id` on create/record (+ `owner_username` via `LEFT JOIN users`), `list_jobs`, `delete_by_owner`, `claim_orphans`.
4. `auth/deps.py` final signatures + stub bodies. Empty `routes_auth.py`/`routes_admin.py` routers (not yet included in `main.py`).
5. Test infra: scrypt-cost autouse fixture, `auth_helpers.py`, `test_db_migrate.py`, jobstore tests.
6. Web keep-green shims.

**DoD:** `uv run pytest` green (665 + new), `make contract-check`, `make lint`, `make test-web` green, `make eval` SUITE PASS. Manual check: copy the real `data/mimic.db` to a temp dir, run `migrate` against it, and confirm the backup is written + `user_version = 1` + all rows readable via the API's `get`.

### P1 — three parallel tracks (no file overlap)

**B1 — auth core (L).** passwords → sessions → users store → limiter → csrf → deps bodies → routes_auth → routes_admin → main wiring → CLI → Makefile → tests.
DoD: B1 tests green; full `pytest` green (B2's suite uses overrides, so B1's wiring must not break it); `curl` flow on a scratch data dir: `make create-admin` → login → `me` → logout; `Origin: http://evil.localhost:5555` POST → 403.

**B2 — authz on existing routes (M).** `get_job` visibility, owner on create, `GET /api/jobs`, health subset, interpreter admin-only, runner guard, migrate `test_api.py`.
DoD: `test_api.py` passes with `as_user` overrides (admin default, plus user/other-user cases); runner guard test; full `pytest` green; `make eval` untouched.

**W-lib (M).** §8.2 + §8.3 + tests + README.
DoD: `pnpm typecheck && pnpm lint && pnpm test` green; README "Auth" section complete enough that W-ui needs no backend.

### P2 — UI + backend matrix (parallel)

**W-ui (L).** §8.1 pages/components in mock mode first, then against the real API. ui-ux-master self-reviews (states, keyboard, focus management in dialogs, the temporary-password reveal, no layout flash in `AuthGate`).
DoD: web tests/typecheck/lint/build green; every state in §8.1 reachable in mock mode (seeded users + `?fail=`).

**INT-backend (S).** `test_authz_matrix.py` with real login across B1 + B2.
DoD: matrix green; full `pytest` + `make eval` + `contract-check` + `lint` green.

### P3 — Integration E2E + docs (S–M, 1 coder)
- Real stack (`uvicorn` without reload + `pnpm dev`) on a **copy** of the pre-auth `data/` with existing jobs: startup migrates + backup; `make create-admin` claims the legacy jobs; admin sees them; create user `alice` → temp password → forced change → upload → alice sees only her job; another user gets 404 on alice's job URL, video and keyframe URLs; admin sees all with the owner filter; disable alice while she is polling → her next poll 401 → login page; reset → forced change; delete alice with a running job → no orphan dir; last-admin guards in the UI; `Test connection` hidden for users and 403 via curl; `127.0.0.1` vs `localhost` mismatch shows the `cookie_rejected` guidance.
- Docs: README, `.env.example`, `web/.env.local.example`, PLAN.md addendum.

**DoD (feature):** everything above, plus `make test`, `make eval`, `make contract-check`, `make lint` green; no credentials, temporary passwords or tokens in logs (grep the uvicorn log of the E2E run for the temp password strings used).

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Host-scoped cookies: every `localhost:*` server the browser talks to receives `mimic_session` | HttpOnly, expiry, documented. Deploying beyond localhost → put web and API on one site over HTTPS with `MIMIC_COOKIE_SECURE=1` |
| `localhost` vs `127.0.0.1` (or different domains) → cookie not sent, login "doesn't stick" | `cookie_rejected` detection + guidance. README + `.env` examples say to use the same hostname |
| Contract change breaks web typecheck mid-flight | P0 keep-green shims; tracks start from a green tree |
| Migration on a live DB (API running during the CLI) | `BEGIN IMMEDIATE` + version re-check inside the transaction, backup first, WAL |
| `make clean-jobs` habit wipes accounts | Retargeted to keep users (A16); full wipe is the explicitly named `reset-data` |
| Deleted user's running job re-creates its dir | Runner deleted-job guard (A14) |
| scrypt memory/CPU under a login burst | Semaphore (4) + rate limiter + threadpool |
| The test suite slows down (scrypt) | Autouse low-cost fixture; one real-cost test |
| In-memory rate limiter resets on restart / per process | Acceptable for a single-process local tool; documented. Multiple uvicorn workers would need a DB counter (not planned) |
| Admin re-running AI labeling on a user's job sends that user's keyframes to the interpreter | Admin action is explicit (U3 grants admins every job action); the existing opt-in disclosure on the button stays |
| Mock "any password" mistaken for real behavior | Mock badge + README + login page mock note |
| Old code started against a v1 DB creates ownerless jobs | Treated as admin-only (A10); not supported, documented |

---

## 13. Open questions (resolved by the user, 2026-10-03)

1. Session lifetimes: **idle 12 h, absolute 7 days** (A1) — confirmed.
2. Temporary passwords: **always server-generated, shown once** (A5) — confirmed.
3. **Users can delete their own jobs — IN SCOPE** (user decision). Admins can delete any job. See §14.

## 14. Addendum — job deletion (user decision 2026-10-03)

- `DELETE /api/jobs/{id}` → 204. Authz: owner or admin; anyone else gets **404** (same non-disclosure rule as reads). Unauthenticated → 401. Origin check applies (unsafe method).
- Queued or running job: mark it deleted and remove row + files; the runner's deleted-job guard (A14) handles a job deleted mid-run (no status writes, dir removed again at the end). Re-run interpretation in progress counts as running → same path. No 409.
- Implementation: `JobStore.delete(job_id)` + `storage.delete_job(job_id)` after commit (best effort, failures logged); audit log line `user <id> deleted job <id>` (ids only).
- Ownership: **B2** (route in `routes_jobs.py` + tests in `test_api.py` / authz matrix), **W-lib** (`deleteJob(jobId)` in `api.ts` + mock + README), **W-ui** (delete action on "My jobs" rows and on the result page, confirmation dialog naming the file, redirect to `/jobs` after deleting the open job).
- Tests: owner deletes own job (files gone), other user → 404 and files intact, admin deletes any job, delete while running → no orphan dir, deleted job → subsequent GET/video/keyframe 404.

---

## P0 notes (2026-10-03)

P0 is done. The code is authoritative: `api/app/auth/{policy,models,deps}.py`, `api/app/core/{db,jobstore,errors}.py`, `api/app/api/schemas.py`, `api/app/config.py`, `api/tests/auth_helpers.py`. These are the gaps and contradictions in the plan, and how P0 resolved them:

**Contract**
- `UserRole` is a named TS literal. The Python enum is `app.auth.models.Role` (the IR already has a TS `Role`, so `UserRole` is the only name that can be exported).
- The request bodies (`LoginRequest`, `ChangePasswordRequest`, `CreateUserRequest`, `UpdateUserRequest`) go into the new `schemas.CONTRACT_REQUEST_MODELS` and are exported in **validation** mode, so fields with a server default are optional in TS (`CreateUserRequest.role?`, `UpdateUserRequest.role?/is_active?`). Response models stay in serialization mode. No existing schema def changed.
- `LoginRequest.username` is normalized (strip + lowercase) inside the model. `CreateUserRequest.username` is normalized *before* the pattern check, so `" Alice "` is stored as `alice` instead of returning 422.
- `Health.interpreter` / `limits` default to `None`, so the anonymous body is `Health(version=..., ffmpeg=...)`. `JobStatus.owner` defaults to `None`.
- The §14 job deletion needs no new error code or schema: it returns 204, or 404 `not_found` for other users.
- `PipelineError` gains a keyword `headers` (for `Retry-After` on `too_many_attempts`). **B1:** `main._pipeline_error` must copy `exc.headers` onto the response, and B1 also owns the `_HTTP_STATUS_CODES` additions (401/403/429), because `main.py` is a B1 file.

**Data layer**
- `core/db.py`: `connect(path, *, timeout_s, autocommit)`, `connection(path)` (implicit transaction, commit/rollback), `transaction(path)` (`BEGIN IMMEDIATE`), and `migrate(path) -> MigrationResult(from_version, to_version, backup_path)`. `MIGRATIONS[i]` upgrades version i to i+1. The backup is written with `journal_mode=DELETE`, so the `.bak` is one self-contained file. Name collisions within the same second get a `-N` suffix.
- **Duplicate removed:** §7 gave `users.claim_orphan_jobs` and §11 gave `JobStore.claim_orphans`, and the user delete needs the job deletion to run inside *its own* transaction. Resolution: `app.core.jobstore` exposes connection-level helpers `delete_jobs_of_owner(conn, owner_id) -> list[str]` and `claim_orphan_jobs(conn, owner_id) -> int`. `SqliteJobStore.delete_by_owner` / `claim_orphans` wrap them in their own transaction. B1's `users.delete` and the CLI bootstrap call the helpers on their `db.transaction` connection. `UserStore` has no `claim_orphan_jobs`.
- `JobStore.create(..., owner_id: str | None = None)` is keyword-optional, so `routes_jobs.py` keeps working until B2 passes the owner. `JobRecord` gains `owner_id` and `owner_username` (via `LEFT JOIN users`).
- `list_jobs(*, owner_id=None, status=None, limit=50, cursor=None) -> JobPage(items, next_cursor)`. `owner_id=None` means every job, including legacy ones. It raises `InvalidCursor` (a `ValueError`) for a bad cursor and `ValueError` for a limit outside 1..200. **B2** maps both to 422 `invalid_request`. The cursor helpers are `encode_cursor` / `decode_cursor`.
- New timestamps are stored with fixed microsecond precision. Legacy rows written by `isoformat()` without microseconds still sort correctly; a test covers it.
- `JobStore.delete(job_id) -> bool` (§14). Writes to a deleted job are silent no-ops, which the runner guard (A14) relies on. There is a test for this.

**Auth package**
- The `AuthServices` fields are typed by the Protocols `UserStore`, `SessionStore` and `AttemptLimiter` in `auth/models.py`, which also has `UserRecord` / `SessionRecord`, `USER_COLUMNS` and `user_from_row`. `password_hash` and `token_hash` are excluded from `repr`. After P0, `app/auth/*` belongs to B1, which may refine the Protocols. Nothing outside `app/auth` uses them except `tests/auth_helpers.py`.
- `AttemptLimiter`: `retry_after(ip, key) -> int | None`, `record_failure(ip, key)`, `reset(ip, key)`. The key is the normalized username, or `"user:<id>"` for password changes.
- `deps.py`: only `get_session_user` (always 401) and `get_optional_user` (always None) are stubs. `get_current_user` (403 `password_change_required`) and `require_admin` (403 `forbidden`) are already final. `get_auth` returns 500 `internal_error` when `app.state.auth` is missing. `SESSION_COOKIE = "mimic_session"` and `session_token(request)` are defined there too.
- The scrypt cost is `policy.ScryptParams` / `policy.SCRYPT_PARAMS` (P0, so that the P0 autouse fixture can lower it before `passwords.py` exists). **B1** must read it through `policy.current_scrypt_params()` at call time, never copy it into a module constant. `@pytest.mark.real_scrypt` (registered in `conftest.py`) opts a test out of the lowered cost.
- `policy.check_password_policy(pw, username, *, current_password=None)` adds the A4 "new ≠ current" rule. `policy.can_access_job(user, owner_id)` is the shared U3/A10 visibility rule for B2 and the INT matrix.
- `tests/auth_helpers.py`: `insert_user(db_path, username, role="user", *, active, must_change, user_id) -> id` (migrates first; the hash can't be used to log in), `fetch_user`, `make_user` (insert + fetch → `UserRecord`), and `as_user(app, user | None)`. `as_user` overrides `get_session_user` and `get_optional_user`, treats a disabled user as 401 and a forced-change user as having no full access, can be nested, and restores the previous overrides.

**Web keep-green**
- `errors.ts` lists the 13 new codes and maps HTTP 401/403/429 to `unauthenticated` / `forbidden` / `too_many_attempts` when the response has no body. `format.ts` has a `describeError` entry for each new code, with `weak_password` built from `PASSWORD_MIN_LENGTH`. `api.ts` `looksLikeHealth` accepts `interpreter: null`. `mockApi.ts` sets `owner: null`. `UploadScreen`/`UploadOptions` use `interpreter?.available`, and `InterpreterStatus` renders nothing while `interpreter` is null.

**Verification**
- The repo has no `data/mimic.db`. The manual check ran against a **copy** of the pre-auth E2E database (`/tmp/mimic-e2e/mimic.db`, 18 jobs, v0): it wrote the backup `mimic.db.pre-auth-<UTC>.bak` (v0, 18 rows), set `user_version = 1` (WAL), all 18 rows read back through `SqliteJobStore.get` with `owner_id` NULL, `list_jobs` returned all 18, a second run did nothing, and the source file's checksum didn't change.

## P3 notes — integration E2E + docs (2026-10-03)

P3 is done. The E2E ran on the real stack: `uvicorn app.main:app` without reload + `pnpm dev`
(non-mock) against a **copy** of the pre-auth E2E data dir (`/tmp/mimic-e2e`: v0 DB, 18 legacy
jobs, all synthetic videos), driven by headless Chromium over CDP plus `curl` for the API-only
checks. Throwaway accounts (`admin`, `alice`, `bob`); the copy, its logs and every password were
deleted afterwards.

**Checklist (all pass)**
- Startup on the v0 copy: backup `mimic.db.pre-auth-<UTC>.bak` written, migrated v0 → v1,
  `/api/auth/status` → `setup_required: true`, login → 409 `setup_required`.
- `make create-admin USER=admin PASSWORD_STDIN=1` → "Assigned 18 existing jobs to admin"; the
  admin's **All jobs** shows all 18.
- UI: create `alice` → temporary password shown once → first sign-in lands on the forced
  `/account/password` (navigating to `/jobs` stays there) → own password → upload → **My jobs**
  shows only her job; no **Test connection** button for her.
- `bob` gets 404 on alice's status / result / video / keyframe / interpret and DELETE, 401
  anonymous, 403 on `GET /api/jobs` while forced; admin gets 200 everywhere; `?owner=<alice>`
  (API and the UI owner filter) returns only her job; a user asking for another owner → 403.
- Admin disables alice while her job page polls → next poll 401 → `/login?next=/jobs/<id>` with
  "Your session ended"; signing in again → `account_disabled`. Enable + reset (UI) → new
  temporary password → forced change again.
- Delete alice right after she uploads (job `processing/probing`) → 204, her rows, sessions and
  all three job dirs gone; runner log "deleted during the run; status writes skipped, files
  removed"; no dir without a row.
- bob deletes his own job in the UI (dir and row gone); bob deleting the admin's job → 404, files
  intact.
- Own-row menu: demote/disable/reset/delete disabled with reasons; API 409 `last_admin` /
  `self_action_forbidden`; `GET /api/admin/users` as a user → 403;
  `POST /api/interpreter/check` → 403 user / 401 anonymous; foreign `Origin` POST → 403.
- 5 wrong passwords → 429 with `Retry-After` (880 s at that point), exposed via CORS; the login form counts
  down (`14:44`, `14:43`, …) and keeps the button disabled.
- `127.0.0.1` vs `localhost`: see the bug below; with `MIMIC_CORS_ORIGINS` allowing the
  `127.0.0.1` origin, login returns 200 but `/me` 401 → **Sign-in didn't stick** naming both hosts
  (+ the proactive **Hostnames don't match** warning).
- `grep -F` of every password, temporary password and wrong password used against the uvicorn and
  Next logs: no match (with a positive control). Logs carry user ids only.

**Bugs found and fixed**
1. **`127.0.0.1` page with the default CORS said "Server unreachable — check that the API is
   running".** The API refuses the foreign origin, so `/api/auth/status` (login page) or `/me`
   (AuthGate) fails as `network_error` before any cookie is involved, and `cookie_rejected` never
   happens. Fix: `describeApiError(error, { hostnameMismatch })` in `web/lib/format.ts` leads a
   `network_error` with the hostname cause and fix (`hostnameMismatchGuidance`); the login
   status error and AuthGate's session error pass `hostnameMismatch()`. Test:
   `format.test.ts` "network_error on a mismatched hostname names the cause first".
2. **`make reset-data` ignored `MIMIC_DATA_DIR`** (always wiped `<repo>/data`, while the API and
   every CLI target use the configured dir). Fix: `DATA_DIR := $(or $(MIMIC_DATA_DIR),data)`.
   `.env` is still not read by make (README says so). Test: `api/tests/unit/test_makefile.py`.

**Findings, not bugs**
- `pnpm dev` opened on `http://127.0.0.1:3000` never hydrates: Next 16 blocks its dev resources
  for non-`localhost` hosts ("Blocked cross-origin request … allowedDevOrigins"), so the page
  stays at "Checking your session…". The mismatch checks therefore ran against `next start` on
  `127.0.0.1:3001`. Documented in README troubleshooting; no `allowedDevOrigins` added (the
  advice is "use localhost").
- A login whose cookie the browser drops still creates a session row; it expires normally.

**Lib cleanups (requested by W-ui)**
- `ApiError.cookieRejected()` now states only what happened (+ both hostnames); the "same
  hostname" advice lives once, in the `cookie_rejected` guidance.
- The "drop guidance the message already says" filter moved from
  `components/ui/errorCopy.ts` into `describeApiError`; `errorCopy` is now an alias, so every
  UI call site behaves as before.

**Docs**: README (accounts section, roles table, sessions/security, HTTP API table, env vars,
make targets, troubleshooting, screenshots), `.env.example`, `web/.env.local.example`,
`docs/PLAN.md` addendum. Screenshots were retaken on a `next build && next start` instance (no
dev indicator, no CSS injection) with a fresh throwaway data dir; the **Legacy** row in
`jobs-admin.png` is a job whose `owner_id` was set to NULL by hand, since the first admin
claims every real legacy job. The temporary password in `admin-temp-password.png` belongs to
that throwaway instance (replaced by alice's own password seconds later, instance deleted).

**Verification**: api `uv run pytest` 1279 passed / 1 skipped; `make eval` SUITE PASS (17/17,
easing 38/38, bands 20/20 · 22/22 · 4/4); `make contract-check` up to date; `make lint` clean;
web typecheck / lint / test (105) / build pass.
