# web/lib — frontend data layer

Everything the UI needs to talk to the API and display the IR. There is no UI code here. Import with `@/lib/...`.

| File | What |
|---|---|
| `types.ts` | **Generated** (`make contract`). Do not edit. API + IR types and constants (`STAGE_ORDER`, `STAGE_LABELS`, `STAGE_WINDOWS`, `BAND_*`, `UNCERTAIN_BELOW`, `PROPERTY_VALUE_KIND`). |
| `api.ts` | Typed client + mock switch. |
| `errors.ts` | `ApiError`, the only error type the layer throws. |
| `upload.ts` | Client-side file checks (extension, size, duration) + `probeVideoDuration`. |
| `jobPoller.ts` | Framework-free poll loop (`pollJob`). |
| `useJobPolling.ts` / `useVideoUpload.ts` / `useHealth.ts` | React hooks. Call them from `"use client"` components (the hook files don't declare a boundary themselves). |
| `format.ts` | Number/value/easing/confidence formatting, stage steps, enum labels, error copy. |
| `timeline.ts` | IR → timeline model (pure) + geometry helpers. |
| `mock/mockApi.ts` | Fake API (lifecycle + scenarios). `mock/sample-result.json` = synced fixture. |

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
- **Go straight to a result:** `/jobs/3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e` (the fixture `job_id`, exported as `MOCK_SAMPLE_JOB_ID`).
- Unknown job id → `not_found` (404).
- To force a scenario from code, pass `uploadVideo(file, opts, onProgress, { mockScenario: "low_confidence" })`.

## api.ts

```ts
getHealth(signal?): Promise<Health>
checkInterpreter(signal?): Promise<InterpreterCheck>   // POST /api/interpreter/check ("Test connection")
uploadVideo(file, options?: Partial<UploadOptions>, onProgress?: (p: UploadProgress) => void,
            request?: { signal?, mockScenario? }): Promise<JobCreated>
getJob(id, signal?): Promise<JobStatus>
getResult(id, signal?): Promise<ResultEnvelope>        // ApiError code "not_ready" (409) before success
rerunInterpretation(jobId, { useInterpreter = true, signal }?): Promise<JobStatus>  // see "Re-run interpretation"
assetUrl(path)                                         // "/api/jobs/<id>/video" → absolute; data:/blob:/http pass through; null → null
DEFAULT_UPLOAD_OPTIONS = { pixelRatio: "auto", useInterpreter: false }   // AI labeling is OPT-IN
API_BASE_URL, IS_MOCK_API
```

`UploadProgress = { loaded, total | null, fraction: 0..1 | null }`.

**Always pass artifact URLs (`artifacts.video_url`, `keyframes[].url`) through `assetUrl()`.** They are API-relative.

### Errors

Every function rejects with `ApiError { code, message, status: number | null, origin: "http" | "job" | "network" | "client", isTransient }`.
`code` is a backend `ErrorCode` or one of the client codes `network_error`, `aborted`, `invalid_response`.
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

- **Privacy:** this sends the job's keyframes to the server-configured AI interpreter, so only call it from an explicit user action. Label the button accordingly (the upload toggle's disclosure applies). Enable it only if `health.interpreter.available`. Otherwise the re-run still succeeds but falls back to heuristic labels with an `interpretation_fallback` warning.
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
// The AI toggle is enabled only if health?.interpreter.available.
const { check, error, isChecking, run } = useInterpreterCheck();
// run() on "Test connection".
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

## Tests

`pnpm test` runs node:test, with no extra deps. `lib/test/register.mjs` resolves extensionless imports and JSON the same way the bundler does.

- `timeline.test.ts`: the model on the fixture plus edge cases.
- `mock/mockApi.test.ts`: every scenario through the real poller.
- `api.mock-mode.test.ts`: the public client with `NEXT_PUBLIC_API_MOCK=1`.

After the contract fixture changes, run `pnpm sync:fixture`.
