# web/lib — frontend data layer

Everything the UI needs to talk to the API and display the IR. There is no UI code here. Import with `@/lib/...`.

| File | What |
|---|---|
| `types.ts` | **Generated** (`make contract`). Do not edit. API + IR types and constants (`STAGE_ORDER`, `STAGE_LABELS`, `STAGE_WINDOWS`, `BAND_*`, `UNCERTAIN_BELOW`, `PROPERTY_VALUE_KIND`). |
| `api.ts` | Typed client + mock switch. Sends the session cookie on every request; broadcasts auth events. |
| `errors.ts` | `ApiError`, the only error type the layer throws. |
| `session.ts` | Session store + `useSession()` (who is signed in), `login` / `logout` / `changePassword`. See **Auth**. |
| `routes.ts` | Pure route policy for `AuthGate`: `routeDecision`, `safeNext`, `loginPath`, `afterAuthPath`. |
| `authPolicy.ts` | Client-side username/password checks (mirror of the server policy) for instant form feedback. |
| `jobList.ts` / `useJobList.ts` | Job list ("My jobs" / "All jobs"): pages, auto-refresh, delete. See **Jobs list & delete**. |
| `adminUsers.ts` / `useAdminUsers.ts` | `/admin/users`: list, mutations, row-action guards. See **Admin**. |
| `upload.ts` | Client-side file checks (extension, size, duration) + `probeVideoDuration`. |
| `jobPoller.ts` | Framework-free poll loop (`pollJob`). |
| `useJobPolling.ts` / `useVideoUpload.ts` / `useHealth.ts` | React hooks. Call them from `"use client"` components (the hook files don't declare a boundary themselves). |
| `format.ts` | Number/value/easing/confidence formatting, stage steps, enum labels, error copy. |
| `timeline.ts` | IR → timeline model (pure) + geometry helpers. |
| `mock/mockApi.ts` | Fake API (lifecycle + scenarios, job ownership, list, delete). `mock/sample-result.json` = synced fixture. |
| `mock/mockAuth.ts` | Fake accounts + session + admin endpoints (**mock only, not security**). |
| `mock/mockJobsRegistry.ts` / `mock/mockStore.ts` | Mock job owners/deletions + mock persistence (localStorage `mimic.mock.*`). |

## Mock mode

`web/.env.local`:

```
NEXT_PUBLIC_API_MOCK=1
# optional: a video to show when the result has no blob URL (e.g. after a reload). Put it in web/public/.
# NEXT_PUBLIC_API_MOCK_VIDEO_URL=/mock-preview.mp4
```

`NEXT_PUBLIC_*` values are baked in when the bundle is built, so restart `pnpm dev` after changing them. In mock mode:

- `uploadVideo` simulates upload progress. It then creates a job that runs `queued → probing → … → generating → done` in about 6 s (AI on) or about 5 s (AI off), and finishes with the fixture. The preview video is a `blob:` URL of the file you uploaded. Keyframes are SVG placeholders that show the element boxes.
- Choose a scenario with **query params on the upload page** (`/`). The scenario is encoded in the job id, so `/jobs/<id>` keeps working after a reload.
  - `?fail=<code>`, **upload-time** (upload rejects with an HTTP status): `unsupported_format` 415, `file_too_large` 413 (rejects at 40 %), `too_long`, `too_short`, `decode_failed`, `invalid_request` 422
  - `?fail=<code>`, **pipeline-time** (the job fails mid-run and keeps the failing stage): `no_motion_detected`, `unsupported_motion`, `no_stable_state` (at scanning), `internal_error` (at measuring), `interrupted` (at decoding)
  - `?scenario=low_confidence`: low/medium bands, at least 2 uncertain transitions (< 0.3), unknown trigger, no cursor, AI timeout with heuristic labels, VFR and timestamp warnings
  - `?scenario=no_preview`: `video_url: null` plus a `preview_unavailable` warning
  - `?scenario=forward_only`: reverse not recorded (one lane, `reverse_not_recorded`)
  - `?scenario=slow`: the AI labeling stage takes ~25 s (long-running state)
  - `?scenario=flaky`: every 3rd status poll fails with a network error (`retrying > 0`, "Reconnecting…")
- Interpreter status: `?interpreter=<s>` with `claude_cli` (default), `ok`, `ok_prompt_only`, `unauthorized`, `unreachable`, `model_not_found`, `no_image_support`, `timeout`, `not_configured`. This sets `getHealth().interpreter` (`not_configured` gives `mode:"none"`, `available:false`) and the `checkInterpreter()` result.
- **Go straight to a result:** `/jobs/3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e` (the fixture `job_id`, exported as `MOCK_SAMPLE_JOB_ID`). It belongs to the mock user `user` (admins see it too; `alice` gets 404).
- Unknown job id → `not_found` (404). Another user's job → `not_found` too.
- **Accounts:** every job route needs a mock session, see **Auth → Mock accounts**. Sign in at `/login` as `admin` or `user` with any password.
- To force a scenario from code, pass `uploadVideo(file, opts, onProgress, { mockScenario: "low_confidence" })`.

## api.ts

```ts
getHealth(signal?): Promise<Health>                    // interpreter/limits null when anonymous
checkInterpreter(signal?): Promise<InterpreterCheck>   // POST /api/interpreter/check ("Test connection"), ADMIN ONLY
uploadVideo(file, options?: Partial<UploadOptions>, onProgress?: (p: UploadProgress) => void,
            request?: { signal?, mockScenario? }): Promise<JobCreated>
getJob(id, signal?): Promise<JobStatus>
getResult(id, signal?): Promise<ResultEnvelope>        // ApiError code "not_ready" (409) before success
listJobs({ owner?, status?, limit?, cursor?, signal? }): Promise<JobList>   // see "Jobs list & delete"
deleteJob(jobId, signal?): Promise<void>               // 204; other user's job → not_found
// auth + admin: getAuthStatus, login, logout, getMe, changePassword, listUsers, createUser,
// updateUser, resetUserPassword, deleteUser, onAuthEvent, hostnameMismatch — see "Auth" / "Admin"
rerunInterpretation(jobId, { useInterpreter = true, signal }?): Promise<JobStatus>  // see "Re-run interpretation"
assetUrl(path)                                         // "/api/jobs/<id>/video" → absolute; data:/blob:/http pass through; null → null
DEFAULT_UPLOAD_OPTIONS = { pixelRatio: "auto", useInterpreter: false }   // AI labeling is OPT-IN
API_BASE_URL, IS_MOCK_API
```

`UploadProgress = { loaded, total | null, fraction: 0..1 | null }`.

**Always pass artifact URLs (`artifacts.video_url`, `keyframes[].url`) through `assetUrl()`.** They are API-relative.

### Errors

Every function rejects with `ApiError { code, message, status: number | null, origin: "http" | "job" | "network" | "client", isTransient, retryAfterS, serverMessage }`.
`code` is a backend `ErrorCode` or one of the client codes `network_error`, `aborted`, `invalid_response`, `cookie_rejected`.
`retryAfterS` = seconds from `Retry-After` (429 `too_many_attempts`), else null. `serverMessage` = the backend's own message, or null when it sent none. `isTransient` is true only for network errors, 5xx and 408 (never for auth codes or 429).
Use `describeError(code, serverMessage?)` from `format.ts` to get `{ title, message, guidance[] }` for the error screen. Pass the server message for job failures and upload rejections: it is more specific (actual duration, size limit) and already says what to do. `guidance` holds the fixes as short bullets (PRD §27 recording guidelines for `no_motion_detected`, `too_short`, `no_stable_state`). Backend codes: the upload ones above, `no_motion_detected`, `unsupported_motion`, `no_stable_state`, `internal_error`, `interrupted`, `not_found`, `not_ready`, `already_running`.

### Interpreter (Layer B) status and connection check

The interpreter is configured **only on the server** through env vars. The browser never enters, stores or sends endpoints, keys or credentials. The UI only displays status.

```ts
InterpreterCheck = {
  mode: "claude_cli" | "openai_compat" | "none", ok: boolean, latency_ms: number | null,
  model: string | null, model_listed: boolean | null, supports_images: boolean | null,
  structured_output: "json_schema" | "json_object" | "prompt_only" | null,
  error: { code: "not_configured" | "unauthorized" | "unreachable" | "timeout" | "model_not_found"
               | "no_image_support" | "bad_response", message } | null,
  checked_at: string,
}
```

- A failed check **resolves** with `ok: false` and `error`. Only transport or 5xx problems reject.
- The check can take seconds. Show a pending state.
- `format.ts`: `INTERPRETER_MODE_LABELS`, `STRUCTURED_OUTPUT_LABELS`, `INTERPRETER_CHECK_ERROR_MESSAGES`, `formatInterpreterCheck(check)` (e.g. "Connected · gpt-4o-mini · 420 ms").
- `InterpreterCheck` is generated into `types.ts`; `api.ts` re-exports it plus the aliases `InterpreterMode`, `StructuredOutputMode`, `InterpreterCheckErrorCode`. Import them from `@/lib/api`.

## Re-run interpretation

A succeeded job can be re-labeled without measuring again: AI labeling (Layer B) + IR assembly + the four outputs run on the stored measurements, in a few seconds (plus the interpreter's own latency). Use it for a "Re-run with AI labeling" button on the result page.

```ts
const { result, job, restart } = useJobPolling(id);

async function onRerun() {
  try {
    await rerunInterpretation(id);   // POST /api/jobs/{id}/interpret {use_interpreter: true}
    restart();                       // only AFTER it resolved: the job is queued again
  } catch (e) {
    const err = ApiError.from(e);    // already_running | not_ready | not_found | network_error …
    // show describeError(err.code, err.message)
  }
}
```

- **Privacy:** this sends the job's keyframes to the server-configured AI interpreter, so only call it from an explicit user action. Label the button accordingly (the upload toggle's disclosure applies). Enable it only if `health?.interpreter?.available`. Otherwise the re-run still succeeds but falls back to heuristic labels with an `interpretation_fallback` warning.
- `rerunInterpretation(id, { useInterpreter: false })` goes back to heuristic labels (nothing leaves the machine).
- Lifecycle after the call: `queued` → `interpreting` → `generating` → `succeeded`/`done`. `restart()` resets the hook to `loading` (`result` is `null`) until the new status arrives, then it behaves exactly like after an upload. If you want to keep the old result on screen while it runs, hold on to it in component state. `job.options.use_interpreter` reflects the re-run's choice as soon as it is queued (so `stageLabel(stage, { useInterpreter })` is right). `job.created_at` stays the upload time, so don't use it for the re-run's elapsed time.
- Errors (reject immediately, nothing changes): `already_running` (409, job is queued/processing, e.g. a double click), `not_ready` (409, the job failed, or the result was made by an older version that stored no measurements; the message says "Upload the video again"), `not_found` (404).
- If the re-run itself fails (rare: interpreter problems normally become a fallback result, not a failure), the job goes back to `succeeded` with the **previous** result, and `job.error` is set (`{code, message}`) even though `status` is `succeeded`. `useJobPolling` ignores it (`error` stays `null`), so check `job?.error` after a re-run to show a notice.
- Measurements, keyframes, preview and every number stay the same. Only labels, roles, hierarchy, interaction type (when the AI is confident), relationships and the generated text can change.
- Mock mode: the re-run takes ~2 s with AI on, ~0.75 s with it off, and is kept in memory only (a page reload shows the original result again).

## Hooks

```ts
const { phase, job, status, stage, stageLabel, progress, result, error,
        isLoading, isTerminal, retrying, lastTransientError, restart, retry } = useJobPolling(id);
```

- `phase`: `loading` → `polling` → `fetching_result` → `succeeded` | `failed`. Render Processing until `isTerminal`, then Result (`result`) or Error (`error`).
- `error.origin === "job"` means the pipeline failed (`job.stage` is where). Anything else is a request failure, for example `not_found` or `network_error` after 5 failed retries.
- Polls every 800 ms, then every 1.5 s after 30 s. Transient errors back off 1 s → 2 s → 4 s … up to 10 s. While retrying, `retrying` is above 0. `restart()` polls the same id again from scratch (after `rerunInterpretation`); `retry()` is the same function, named for the error state's "Try again".
- `progress` is 0..1 overall. Elapsed time can be derived from `job.created_at`.

```ts
const { state, progress, error, job, upload, cancel, reset } = useVideoUpload();
// state: idle | uploading | done | error.
// upload() resolves JobCreated | null and never throws.
const created = await upload(file, { pixelRatio, useInterpreter });
if (created) router.push(`/jobs/${created.id}`);
```

```ts
const { health, error, isLoading, refresh } = useHealth();
// The AI toggle is enabled only if health?.interpreter?.available.
// interpreter/limits are null for anonymous callers and during a forced password change.
const { check, error, isChecking, run } = useInterpreterCheck();
// run() on "Test connection" — admins only (users get 403 `forbidden`; hide the button for them).
```

### Client-side upload checks (`upload.ts`)

`checkVideoFile(file, limits?)` checks the extension and the size limit. Limits come from the server: `limitsFromHealth(health)` turns `GET /api/health`'s `limits` into an `UploadLimits` (falls back to `DEFAULT_UPLOAD_LIMITS` = backend defaults while health is loading or unavailable); pass it to both checks. `probeVideoDuration(file)` returns seconds, or `null` if the browser can't decode the file (HEVC .mov). In that case skip the duration check. `checkVideoDuration(s, limits?)` checks the maximum (+ tolerance) and minimum duration. Each check returns `{ ok: true } | { ok: false, code, message }`. Also exported: `ACCEPT_ATTRIBUTE` (for `<input accept>`), `MAX_UPLOAD_MB`, `MAX_DURATION_S`.

## format.ts

- Values (rounding is half away from zero, like the backend generators; `format.test.ts` pins parity): `formatPx` (|v| ≥ 3 → whole px, else 0.5 px), `formatScale` (2 dp), `formatOpacity` (0.05), `formatColor`, `formatShadow`, `formatValue(value, property)`, `formatChange(transition)` ("0px → -8px"), `formatScaleChange` ("+6%"), `formatPercent`, `formatNumber`. These follow the backend phrasing rules, so the UI matches the generated text.
- Time: `formatMs(ms, { approx })` (approx = "~280 ms", rounded to 10), `formatSeconds` ("1.18 s"), `formatElapsed` ("1:15").
- Easing: `formatEasing(easing, { withNearest })`, `formatCubicBezier`, `easingToCss`.
- Confidence: `bandFor`, `BAND_LABELS`, `formatConfidence(c)` ("Medium · 76%"), `isUncertain(overall)` (< 0.3).
- Stages: `PIPELINE_STAGES`, `stageLabel(stage, { useInterpreter })` (drops "(AI)" when AI is off), `stageSteps(job)` → `[{ stage, label, state: done | current | pending | failed }]` for the processing checklist.
- Labels: `PROPERTY_LABELS`, `SEGMENT_LABELS`, `INTERACTION_TYPE_LABELS`, `TRIGGER_LABELS`, `REVERSE_TRIGGER_LABELS`, `DIRECTION_LABELS`, `PATTERN_LABELS`, `ROLE_LABELS`, `ELEMENT_KIND_LABELS`, `INTERPRETATION_STATUS_LABELS`, `WARNING_LABELS`, `formatPixelRatio`.

## timeline.ts — model shape

```ts
const model = buildTimeline(result.spec, { includeUncertain = true, paddingMs? });

TimelineModel {
  window: { start_ms, end_ms }      // absolute video ms covering all lanes and cursor events
  tracks: TimelineTrack[]           // one per element × property, in hierarchy order, shared by all lanes
  lanes: TimelineLane[]             // one per segment, in spec order: fwd, rev | rt_in, rt_out
  markers: TimelineMarker[]         // cursor events: cursor_enter | cursor_leave | cursor_stationary | cursor_move_start
  rowCount, uncertainCount, isEmpty
}
TimelineLane { segment, kind, label ("Forward"/"Reverse"/"Press"/"Release"), window (padded, per lane),
               onset_ms, settle_ms, trigger_event_ms, rows: TimelineRow[], markers (trigger/onset/settle) }
TimelineTrack { key "e1:translateY", element_id, element_label, element_role, depth, parent_id,
                is_target, property, property_label }
TimelineRow extends TimelineTrack { transition_id, segment_id, start_ms, end_ms, duration_ms, delay_ms,
                confidence, confidence_band, uncertain, easing, from, to, delta, transform_origin,
                notes, samples: [t_ms, progress][] }
```

- All times are **absolute video ms**. To seek: `video.currentTime = row.start_ms / 1000`.
- Lanes have their own windows. The forward and reverse runs can be seconds apart, so draw each lane on its own axis. Use `model.window` if you want a single shared axis instead.
- Rows in each lane follow `tracks` order. To align lanes as a grid, iterate `tracks` and look up `lane.rows.find(r => r.key === track.key)`. A lane can be missing a track.
- `uncertain` rows (overall < 0.3) must look different **without relying on color alone**. `includeUncertain: false` hides them. `isEmpty` drives the empty state.
- Helpers: `timeToFraction(t, window)`, `fractionToTime(f, window)` (click-to-seek), `rowSpan(row, window)` → `{ left, width }` as 0..1, `axisTicks(window, n)` (1/2/5 steps), `activeRowsAt(model, t)` (playhead highlight), `laneAt(model, t)`, `findRow(model, transitionId)`.

## Auth

Accounts are admin-managed (no sign-up). The API keeps the session in an **HttpOnly cookie** (`mimic_session`) that JavaScript never sees. The client sends it on every request (`fetch(..., { credentials: "include" })`, the XHR upload with `withCredentials`) and `<video>`/`<img>` send it by themselves (don't add a `crossorigin` attribute). Nothing auth-related is stored in localStorage/sessionStorage, and you never handle tokens.

### Session store (`session.ts`)

```ts
const session = useSession();   // starts loading on first use; "loading" during SSR/prerender
session.status  // "loading" | "anonymous" | "authenticated" | "error"
session.me      // Me { id, username, role: "admin" | "user", must_change_password, created_at } | null
session.error   // ApiError | null — see below
isAdminSession(session)          // admin with a full session

await login(username, password)  // → Me; rejects with ApiError (store stays "anonymous")
await logout()                   // → anonymous; rejects (state unchanged) if the API is unreachable
await changePassword(current, next)  // → Me (must_change_password: false); store updated
refreshSession()                 // re-read /api/auth/me ("Try again" in the error state)
getSession() / subscribeSession(cb) / loadSession()   // non-React access
```

- `error` while `status === "error"`: `/me` failed (API down, 5xx). Show "can't reach the server" + Try again → `refreshSession()`.
- `error` while `status === "anonymous"`: `unauthenticated` when a **signed-in** session ended (expired, admin disabled/deleted the account or changed its role) → "Your session expired, sign in again". null after a normal sign-out or on first visit.
- Any API call that gets 401 `unauthenticated` switches the store to `anonymous` (via `onAuthEvent`). A 403 `password_change_required` flips `me.must_change_password` to true. So job polling, the list, uploads etc. don't need their own auth handling: `AuthGate` re-renders and redirects.
- No React provider is needed (module store + `useSyncExternalStore`).

### Route policy (`routes.ts`)

`AuthGate` (client component in the root layout) does:

```ts
const session = useSession();
const d = routeDecision(usePathname(), session, { search: window.location.search /* in an effect */, next });
// loading → neutral placeholder (never the page) · error → retry state (d.error)
// redirect → router.replace(d.to) · forbidden → 403 state · render → children
```

| Path | Anonymous | Forced change | User | Admin |
|---|---|---|---|---|
| `/login` | render | → `/account/password?next=…` | → `next` or `/` | → `next` or `/` |
| `/account/password` | → `/login?next=/account/password` | render | render | render |
| `/admin`, `/admin/*` | → `/login?next=…` | → `/account/password?next=…` | **forbidden** (not a redirect) | render |
| everything else (`/`, `/jobs`, `/jobs/[id]`) | → `/login?next=<path+query>` | → `/account/password?next=…` | render | render |

- `/login` renders even when the session is in `error` (the form shows network errors itself).
- `safeNext(raw, fallback = "/")`: validate any `?next=` before redirecting. Accepts only same-app paths; rejects `//evil`, `https://…`, backslashes, control chars, encoded variants (`/%2F%2Fevil`) and `/login` (loop). Pass the raw `searchParams.next` (string or array).
- `afterAuthPath(me, next)`: where to go after login / password change (forced change first). `loginPath(next)`, `changePasswordPath(next)` build the URLs.
- Path constants: `LOGIN_PATH`, `CHANGE_PASSWORD_PATH`, `JOBS_PATH`, `ADMIN_USERS_PATH`.

### Login page

```ts
const status = await getAuthStatus();   // { setup_required } — public
if (status.setup_required) → show SETUP_COMMAND ("make create-admin", from format.ts) + "Check again" (call getAuthStatus again). No form.
const mismatch = hostnameMismatch();     // { pageHost, apiHost } | null — warn before the user even tries
try { const me = await login(username, password); router.replace(afterAuthPath(me, next)); }
catch (e) { const err = ApiError.from(e); const d = describeApiError(err); /* … */ }
```

| `err.code` | When | UI |
|---|---|---|
| `invalid_credentials` (401) | wrong username or password | inline form error |
| `account_disabled` (403) | correct password, account disabled | "ask an admin" |
| `too_many_attempts` (429) | 5 failures / 15 min for this username (+ per-IP cap) | countdown from `err.retryAfterS` (`formatRetryAfter`, `formatElapsed`); null → generic "wait a few minutes" |
| `setup_required` (409) | no admin exists | switch to the setup state |
| `cookie_rejected` (client) | login OK but `/me` 401 right after: the browser didn't keep the cookie | `err.message` names both hostnames when known; guidance: same hostname for web and API (both `localhost`, not `127.0.0.1`) |
| `invalid_request` (422) | empty field | (prevent client-side) |
| `origin_not_allowed` (403) | page origin not in `MIMIC_CORS_ORIGINS` | config error |
| `network_error` | API down | retry |

### Change password page (`/account/password`)

Forced variant when `session.me.must_change_password` (explain why, no app nav except sign out), voluntary otherwise. Client checks before submitting: `newPasswordProblems({ newPassword, confirmPassword, username: me.username, currentPassword })` from `authPolicy.ts` → `{ newPassword?, confirmPassword? }` (length `PASSWORD_MIN_LENGTH`..`PASSWORD_MAX_LENGTH` in code points after NFKC, ≠ username, ≠ current, confirm matches). Then `await changePassword(current, next)` from `session.ts` and `router.replace(safeNext(next))`. Errors: `current_password_incorrect` (422, on the current-password field), `weak_password` (422, show `describeApiError(err).message`, it names the failed rule), `too_many_attempts` (429), `unauthenticated` (store goes anonymous by itself). The server signs out every other session of the account and gives this browser a new one.

### Error copy

`describeError(code, serverMessage?)` has an entry for every code incl. the auth ones; `describeApiError(err, { hostnameMismatch? })` additionally prefers `err.serverMessage`, uses the client message for `cookie_rejected` (it says what happened; the advice is the guidance), turns `retryAfterS` into the `too_many_attempts` guidance, leads a `network_error` with the hostname cause when given `hostnameMismatch()` (with the default `MIMIC_CORS_ORIGINS` a `127.0.0.1` page can't reach a `localhost` API at all), and drops guidance lines the message already says. `components/ui/errorCopy` is an alias of it. Also in `format.ts`: `USER_ROLE_LABELS` (`admin` → "Admin"; `ROLE_LABELS` is the IR element roles), `formatLastLogin(iso | null)` ("Never", "5 min ago", "yesterday", "12 Sep 2026"), `formatRelativeTime(iso)`, `formatRetryAfter(s)`, `formatJobOwner(owner)` ("Legacy" for `null`), `SETUP_COMMAND`.

### Hostnames

The cookie is set by the API host and is host-only. `localhost:3000 → localhost:8000` works (ports don't matter). Mixing `127.0.0.1` and `localhost` doesn't: the browser drops the cookie and login "doesn't stick" → `cookie_rejected`. `hostnameMismatch()` compares `window.location.hostname` with `NEXT_PUBLIC_API_BASE_URL` (null in mock mode).

### Mock accounts (`NEXT_PUBLIC_API_MOCK=1`)

> **Mock only, not security.** Any non-empty password signs in, except the literal `wrong` (→ `invalid_credentials`). Show the existing Mock API badge on the login page and say so.

| Username | Role | State |
|---|---|---|
| `admin` | admin | the only admin (last-admin guards apply) |
| `user` | user | owns the fixture job + 1 more |
| `alice` | user | owns 2 jobs (one failed) |
| `newbie` | user | must change password at sign-in |
| `disabled` | user | inactive → `account_disabled` |

Plus one **legacy** job (`owner: null`, admins only). The mock session and accounts persist in localStorage (`mimic.mock.auth`, `mimic.mock.jobs`; no passwords are stored). Reset: clear those keys, or `resetMockData()` from `mock/mockApi`. In the mock: current password `wrong` → `current_password_incorrect`; the password policy, the login throttle (5 × `wrong` → `too_many_attempts` with Retry-After), `last_admin`, `self_action_forbidden`, `username_taken` and session revocation (own role change → signed out) behave like the server. Temporary passwords are random 16-char strings, never stored.

**`?fail=<code>` on the page URL** forces a state (each code only affects the operations listed):

| `?fail=` | Affects | Result |
|---|---|---|
| `setup_required` | `getAuthStatus`, `login` | `{setup_required: true}` / 409 |
| `invalid_credentials`, `account_disabled`, `origin_not_allowed` | `login` | 401 / 403 / 403 |
| `too_many_attempts` | `login`, `changePassword` | 429, `retryAfterS = 90` |
| `cookie_rejected` | `login` | login 200 but no session → `session.login` rejects `cookie_rejected` |
| `network_error` | `login` (every time); `listUsers`, `listJobs` (**first call per page load**, so Try again works) | `network_error` |
| `unauthenticated` | `listUsers`, `listJobs` (first call; also ends the mock session) | 401 → store anonymous ("session expired") |
| `current_password_incorrect`, `weak_password` | `changePassword` | 422 |
| `forbidden` | `listUsers` (first call) | 403 |
| `username_taken` | `createUser` | 409 |
| `last_admin` | `updateUser`, `deleteUser` | 409 |
| `self_action_forbidden` | `updateUser`, `resetUserPassword`, `deleteUser` | 409 |
| `not_found` | `updateUser`, `resetUserPassword`, `deleteUser`, `deleteJob` | 404 |

These don't collide with the upload page's `?fail=` codes. Tests/dev code can sign in directly with `mockSignInAs("admin" | … | null)` from `mock/mockAuth`.

## Admin

`/admin/users` (admins; `routeDecision` gives `forbidden` to users).

```ts
const { users, isLoading, isRefreshing, error, refresh,
        createUser, updateUser, resetPassword, deleteUser } = useAdminUsers();

users: AdminUser[]  // { id, username, role, is_active, must_change_password, created_at, updated_at,
                    //   last_login_at | null, job_count }, sorted by username
const r = await createUser({ username, role });      // MutationResult<TemporaryPassword>
if (r.ok) reveal(r.data.temporary_password);         // shown ONCE — keep it in component state only
else show(describeApiError(r.error));                // username_taken (409), invalid_request (422, bad username)
await updateUser(id, { role: "admin" });             // MutationResult<AdminUser>
await updateUser(id, { isActive: false });           // disable (sessions revoked) / true = enable
await resetPassword(id);                             // MutationResult<TemporaryPassword>, forces a change
await deleteUser(id);                                // MutationResult<void>: account + ALL its jobs/files
```

- Mutations never throw: `{ ok: true, data } | { ok: false, error: ApiError }`. After a 404/409 the list reloads itself (someone else changed something).
- Row actions: `userActionGuards(user, users, session.me.id)` → `{ isSelf, isLastActiveAdmin, can: { promote, demote, disable, enable, resetPassword, delete }, reasons, roleChangeSignsOut }`. Own row: no disable/reset/delete. Last active admin: no demote/disable/delete, `reasons.*` gives the text for a disabled control. Still handle `last_admin` / `self_action_forbidden` from the server.
- Changing **your own role** revokes your sessions: confirm first (`roleChangeSignsOut`); afterwards the hook refreshes the session store, which goes `anonymous` → login page.
- Delete dialog: state `user.job_count` and require typing the username: `deleteConfirmationMatches(typed, user.username)`.
- Client-side username check for the create dialog: `usernameProblem(raw)` / `USERNAME_HINT` (`authPolicy.ts`); the server normalizes `" Alice "` → `alice`.
- Owner filter options for the jobs page: `listUsers()` (admins only).

## Jobs list & delete

```ts
const { items, isLoading, isRefreshing, error, hasMore, isLoadingMore, loadMoreError,
        hasActiveJobs, refresh, loadMore, remove } = useJobList({ owner, status, pageSize });
```

- `owner`: omitted = users see their own jobs, admins see **all** (incl. legacy `owner: null`); `"me"` = own; a user id = that user's jobs (admin only, else 403 `forbidden`). `status`: `JobState` filter. `pageSize` default 50 (max 200). Changing any of them starts a fresh list.
- `items: JobStatus[]`, newest first. Each has `owner: { id, username } | null` (`formatJobOwner` → "Legacy"), `source?.filename` (null until probed), `status`/`stage`/`progress`, `options.use_interpreter`, `created_at`. Link rows to `/jobs/<id>`.
- States: `isLoading` (first load) · `error` with no items → error + Try again (`refresh()`) · `error` with items → keep rows, show a banner · empty → CTA to upload. `hasMore` → "Load more" (`loadMore()`, errors in `loadMoreError`).
- Refreshes every 5 s while `hasActiveJobs`, and on window focus. A refresh re-reads every row on screen in one request, so loaded pages don't vanish.
- `remove(jobId)` → `DELETE /api/jobs/{id}`, drops the row, resolves `null` (also for `not_found`: already gone) or the `ApiError` to show.
- Outside the list (result page): `await deleteJob(jobId)` from `api.ts`, then `router.push("/jobs")`. Owner or admin may delete; anyone else gets `not_found`. A queued/running job can be deleted (the server stops it writing). Afterwards every read of the job, its video and keyframes is 404.
- `/jobs/[id]` for another user's job is `not_found` (never 403); `describeError("not_found")` says "doesn't exist, or you don't have access". Admins viewing someone else's job see `job.owner`.
- Framework-free: `new JobListController(query, { client, refreshIntervalMs })` (tests / non-React).

Mock: `listJobs` pages with an offset cursor (`o:<n>`, opaque); a bad cursor → 422 `invalid_request`. Uploads are owned by the signed-in mock user and persist across reloads (the preview video doesn't). Deleted ids stay 404.

## Tests

`pnpm test` runs node:test, with no extra deps. `lib/test/register.mjs` resolves extensionless imports and JSON the same way the bundler does.

- `timeline.test.ts`: the model on the fixture plus edge cases.
- `mock/mockApi.test.ts`: every scenario through the real poller.
- `api.mock-mode.test.ts`: the public client with `NEXT_PUBLIC_API_MOCK=1` (lifecycle, ownership/U3, list filters, delete, health subset, admin-only check).
- `api.test.ts`: the real client against a stubbed `fetch`/XHR (credentials, methods, 204, Retry-After, auth events, `cookie_rejected`, session error state).
- `session.test.ts`, `routes.test.ts`, `jobList.test.ts`, `adminUsers.test.ts`, `mock/mockAuth.test.ts`: session store, route policy + `safeNext`, list controller, admin guards, mock guards and every `?fail=` scenario.

After the contract fixture changes, run `pnpm sync:fixture`.
