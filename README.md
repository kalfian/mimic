# mimic

Mimic takes a short screen recording of a single UI interaction (hover, click, dropdown, modal,
accordion, ...) and gives you a motion spec: what moved, by how much, when, and with which
easing. The numbers come from measuring the video's pixels with OpenCV, so they are estimates,
not the original CSS. Optional AI labeling names the elements. The result comes in four
formats: an **LLM prompt** you can paste into a coding agent, a technical spec, the JSON spec
(called the IR, Mimic's intermediate representation) and suggested CSS.

![Result page: video preview, detected interaction, motion timeline](docs/screenshots/result.png)

Contents: [Prerequisites](#prerequisites) · [Setup](#setup) · [Run](#run) ·
[How to use](#how-to-use) · [AI labeling](#ai-labeling-optional) · [CLI](#cli-analysis-without-the-server) ·
[Tests & eval](#tests-and-synthetic-evaluation) · [Make targets](#make-targets) ·
[Debugging](#debugging-mimic_debug) · [Troubleshooting](#troubleshooting) ·
[Architecture](#architecture) · [Limitations](#known-limitations)

## Prerequisites

| Tool | Version | Why | Install (macOS) | Verify |
|---|---|---|---|---|
| [uv](https://docs.astral.sh/uv/) | recent (0.11 tested) | Python env + deps for `api/`. Installs **Python 3.13** itself (pinned in `api/.python-version`), so you don't need a separate Python install | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| [Node.js](https://nodejs.org/) | **≥ 22.13** (pnpm 11 needs it; Next.js 16 alone needs ≥ 20.9) | Frontend (`web/`) | `brew install node` (or nvm / asdf) | `node --version` |
| [pnpm](https://pnpm.io/) | **11.x** (`packageManager: pnpm@11.15.1`) | Frontend package manager | `brew install pnpm`, `npm install -g pnpm@11` or `curl -fsSL https://get.pnpm.io/install.sh \| sh -` | `pnpm --version` |
| [FFmpeg](https://ffmpeg.org/) (`ffmpeg` + `ffprobe`) | any recent (8.1 tested) | Probing, decoding and the preview video | `brew install ffmpeg` | `ffmpeg -version`, `ffprobe -version` |
| [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) CLI | optional | Default AI labeling backend (`MIMIC_INTERPRETER=claude_cli`), uses your Claude login | `curl -fsSL https://claude.ai/install.sh \| bash` or `npm install -g @anthropic-ai/claude-code`, then run `claude` once to sign in | `claude --version` |
| OpenAI-compatible gateway (e.g. 9Router) | optional | Alternative AI labeling backend (`openai_compat`); the model must accept images | see the gateway's docs | "Test connection" in the UI |

**Linux:** use the official installers above for uv, pnpm and Claude Code. Install Node.js 22+
with nvm or your distro's NodeSource packages, and FFmpeg with your package manager
(`sudo apt install ffmpeg`, `sudo dnf install ffmpeg`). Everything else works the same.

`make check-tools` checks that `uv`, `pnpm`, `ffmpeg` and `ffprobe` are on `PATH`. If `claude`
is missing it prints a note, because AI labeling is optional.

## Setup

```bash
git clone https://github.com/kalfian/mimic.git
cd mimic
make setup                                  # check-tools → uv sync (api/) → pnpm install (web/) → data/jobs/
cp .env.example .env                        # backend settings (optional, all values have defaults)
cp web/.env.local.example web/.env.local    # frontend settings (optional, defaults shown below)
```

`make setup` ends by printing the OpenCV and NumPy versions. If that line appears, the backend
is ready.

**Backend `.env`** (repo root, gitignored, read by `api/app/config.py`, prefix `MIMIC_`):

| Variable | Default | What it does |
|---|---|---|
| `MIMIC_DATA_DIR` | `data` | Where the job DB (`mimic.db`) and `jobs/<id>/` (upload, preview, keyframes, results) go |
| `MIMIC_MAX_UPLOAD_MB` / `MIMIC_MAX_DURATION_S` / `MIMIC_MIN_DURATION_S` | `200` / `15.0` / `0.5` | Upload limits (the UI reads them from `/api/health`) |
| `MIMIC_INTERPRETER` | `claude_cli` | AI labeling backend: `claude_cli`, `openai_compat` or `none` (see [AI labeling](#ai-labeling-optional)) |
| `MIMIC_CLAUDE_MODEL` | `sonnet` | Model alias passed to `claude --model` |
| `MIMIC_LLM_BASE_URL` / `MIMIC_LLM_API_KEY` / `MIMIC_LLM_MODEL` | unset | Only for `openai_compat` |
| `MIMIC_CORS_ORIGINS` | `http://localhost:3000` | Browser origins allowed to call the API |
| `MIMIC_FFMPEG_BIN` / `MIMIC_FFPROBE_BIN` | `ffmpeg` / `ffprobe` | Set these if FFmpeg is not on `PATH` |
| `MIMIC_DEBUG` | `0` | `1` writes per-job debug dumps (see [Debugging](#debugging-mimic_debug)) |

**Frontend `web/.env.local`** (Next.js only reads env files from `web/`):

| Variable | Default | What it does |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | The API base URL. The browser calls it directly |
| `NEXT_PUBLIC_API_MOCK` | `0` | `1` runs the UI against a fake in-browser API and a fixture result, so you don't need a backend |

Secrets such as `MIMIC_LLM_API_KEY` go only in your local `.env`. Never put them in the
`*.example` files. `NEXT_PUBLIC_*` values are built into the frontend bundle, so restart
`pnpm dev` after you change them.

## Run

Use two terminals:

```bash
make dev-api      # FastAPI on http://localhost:8000 (uvicorn --reload)
make dev-web      # Next.js on http://localhost:3000
```

Then open **http://localhost:3000**. `http://localhost:8000/api/health` should report
`"ffmpeg": true`.

`make dev-api` restarts on every file edit under `api/`. A restart interrupts any running job,
and that job ends as `failed` / `interrupted`. When you are only using the app, run the API
without reload:

```bash
cd api && uv run uvicorn app.main:app --port 8000
```

**Frontend-only work:** set `NEXT_PUBLIC_API_MOCK=1` in `web/.env.local` and run only
`make dev-web`. You get a "Mock API" badge, a simulated job lifecycle and the fixture result. You
can also go straight to the fixture result at
`/jobs/3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e`. The scenario query params (`?fail=…`,
`?scenario=…`, `?interpreter=…`) are listed in [`web/lib/README.md`](web/lib/README.md#mock-mode).

## How to use

### 1. Record a good clip

How well the analysis works depends mostly on the recording. On the upload page, the right-hand
panel shows the same checklist:

1. Keep it short: **2–15 s** (the hard limits are 0.5–15 s and 200 MB). Formats: MP4, MOV, M4V, WebM.
2. Record at **60 fps** if you can, with the browser at **100 % zoom**. Keep the cursor visible.
3. Start recording, **wait about 1 s** without touching anything, then perform **one** interaction.
4. Wait until the animation has finished. For hovers, move the pointer away and let the reverse
   animation finish too: wait → enter → animate → hold → leave → reverse → wait.
5. Record one component. Don't scroll or change pages during the recording.

On macOS, use **Cmd + Shift + 5** → "Record Selected Portion" (or QuickTime Player → File → New
Screen Recording). Under Options, turn on "Show Mouse Clicks", and trim the clip in QuickTime
(Cmd + T) if it runs longer than 15 s. A Retina capture gives a large video, for example
2880 × 1800. Upload it as **2x** (see below).

### 2. Upload and choose options

![Upload page](docs/screenshots/upload.png)

Drop the file or click "choose a file". The page reads the duration in the browser and checks it
against the server's limits before uploading.

![Upload page with a file selected](docs/screenshots/upload-file-selected.png)

- **Display scale.** Video pixels ÷ scale = CSS px. **Auto** guesses 2x for large captures
  (≥ 2560 px wide or ≥ 1600 px tall). If you know the screen, choose `1x` / `2x` (Retina) / `3x`
  (phone). A wrong scale halves or doubles every distance.
- **AI labeling (opt-in, off by default).** When it is on, a few keyframe stills (start, middle
  and end states) go to the configured interpreter so it can name elements ("Product image"
  instead of "Image (e2)"). **Don't turn it on for confidential screens.** With it off, nothing
  from the recording leaves the API server. Every number is measured from the video either way.
  The AI never changes a measured value.

Click **Analyze recording**. It usually takes 5–15 s. AI labeling can add up to about 2 minutes.

### 3. Processing

![Processing page with stage list](docs/screenshots/processing.png)

The job page lists the pipeline stages with their live status. You can reload it or come back
later with the same URL (`/jobs/<id>`). If the analysis fails, the page explains what went wrong
and what to try next (see [Troubleshooting](#troubleshooting)).

### 4. Read the result

The top of the result page (hero image above) has:

- **Video player** with frame stepping, a scrubber and 0.25× / 0.5× / 1× speed. It stays in sync
  with the timeline playhead.
- **Detected interaction:** type (hover, click, open/close, ...), trigger, target element,
  forward/reverse durations, direction and display scale. Each item has a confidence badge.
- **Notes on this analysis:** warnings that tell you how far to trust the result, for example
  VFR video, a missing reverse, `frames_subsampled` or heuristic labels.

![Motion timeline with a selected transition and the inspector](docs/screenshots/timeline-inspector.png)

**Motion timeline.** There is one row per element property and one column per motion (forward /
reverse). Each bar shows when a transition starts and how long it takes, with its easing curve
drawn inside. Click a bar, or move with the keyboard, to open the inspector below it. The
inspector shows from → to values, duration, delay, easing (the family plus a
`cubic-bezier(...)`), origin, the fitted curve against the measured points, and a confidence
breakdown (value / timing / easing).

- Keyboard: arrow keys move between bars, **Enter** / **Space** selects one, **PgUp/PgDn** jump
  between elements, and **Home/End** go to the first/last column. On the time axes, ←/→ moves the
  playhead by one frame and Shift + arrow moves it 50 ms.
- Confidence: **High** ≥ 0.8, **Medium** ≥ 0.5, **Low** < 0.5 (hatched). Changes below 0.3 are
  **uncertain** (dashed outline). They are shown by default, and a checkbox hides them.

Under the timeline are the **keyframes** (A · start, 25 / 50 / 75 %, B · end, with optional
element boxes) and before/after crops of each changed element.

### 5. Copy the output

![Motion specification tabs, LLM Prompt selected](docs/screenshots/spec-llm-prompt.png)

**Copy LLM prompt** (top right) or the **Motion specification** tabs give you:

| Tab | Use it for |
|---|---|
| **LLM Prompt** (default) | Paste into a coding LLM or agent, ideally with the recording attached |
| Technical | A readable spec for a developer |
| JSON | The motion IR (schema 0.1, no per-frame samples) |
| CSS | A suggested implementation. The original site may have used other CSS, JS or an animation library |

### 6. Add AI labels afterwards

If the job ran with labeling off, the "Labels are heuristic" note has a **Label with AI**
button. It re-runs only the naming step, using the stored measurements
(`POST /api/jobs/{id}/interpret`). The video is not decoded again and no measured numbers
change. The prompt, JSON and CSS are regenerated. This button is the explicit opt-in: clicking it
sends the keyframes to the interpreter. If the re-run fails, the previous result stays.

Dark theme follows the OS (`prefers-color-scheme`):

![Result page, dark theme](docs/screenshots/result-dark.png)

## AI labeling (optional)

The backend is configured on the server only, in `.env`. The browser never sees keys or tokens.

| `MIMIC_INTERPRETER` | What runs when a job opts in | Settings |
|---|---|---|
| `claude_cli` (default) | A `claude -p` subprocess that uses the Claude Code login under `$HOME` (no API key) | `MIMIC_CLAUDE_BIN` (`claude`), `MIMIC_CLAUDE_MODEL` (`sonnet`), `MIMIC_CLAUDE_TIMEOUT_S` (`120`) |
| `openai_compat` | Any OpenAI-compatible gateway (`POST {base}/chat/completions`). Keyframes are sent as images, so use a vision model | `MIMIC_LLM_BASE_URL`, `MIMIC_LLM_API_KEY`, `MIMIC_LLM_MODEL`, `MIMIC_LLM_TIMEOUT_S` (`120`), `MIMIC_LLM_SUPPORTS_JSON_SCHEMA` (`auto`) |
| `none` | Nothing. Elements get heuristic labels only | none |

- **claude_cli:** install Claude Code, run `claude` once on the API host to sign in, and keep
  the default `.env`.
- **openai_compat**, for example with a local 9Router (the values are placeholders):

  ```bash
  MIMIC_INTERPRETER=openai_compat
  MIMIC_LLM_BASE_URL=http://localhost:20128/v1
  MIMIC_LLM_API_KEY=<your-gateway-token>
  MIMIC_LLM_MODEL=<model-id-that-accepts-images>
  ```

Restart the API after you change `.env`.

**Check the connection.** The upload page shows the interpreter mode, model and status. Click
**Test connection** to run `POST /api/interpreter/check` from the API server:

![Interpreter status and Test connection result](docs/screenshots/interpreter-test-connection.png)

```bash
curl -s -X POST http://localhost:8000/api/interpreter/check
# {"mode":"claude_cli","ok":true,"latency_ms":86,"model":"sonnet",...,"error":null,...}
```

- `openai_compat`: calls `GET {base}/models`, then sends one tiny chat completion with a
  generated test image (never your data). It reports whether the model is listed, whether it
  accepts images and which structured-output mode works.
- `claude_cli`: checks only that the binary exists and that `claude --version` runs. It does
  **not** prove that you are signed in. If you aren't, labeling jobs fall back to heuristic
  labels and the result says so.
- Failure codes include `not_configured`, `unauthorized`, `unreachable`, `timeout`,
  `model_not_found`, `no_image_support` and `bad_response`. The UI shows what to fix for each one.

`GET /api/health` reports whether the mode is configured, without calling it. An interpreter
failure (timeout, error, bad output) never fails a job: the job falls back to heuristic labels
and adds a warning. Tokens are never logged, sent to the browser or written to job files.

## CLI analysis (without the server)

```bash
cd api
uv run python scripts/analyze.py rec.mov --pixel-ratio 2               # summary + stage timings
uv run python scripts/analyze.py rec.mp4 --print llm_prompt            # one output: technical|llm_prompt|css|json|ir
uv run python scripts/analyze.py rec.mp4 --out result.json --keyframes kf/ --debug dbg/
uv run python scripts/analyze.py rec.mp4 --use-interpreter             # AI labels (needs MIMIC_INTERPRETER)
```

If the recording can't be analysed, the script prints the API error
(e.g. `error: no_motion_detected: …`) and exits with code 3.
`uv run python -m app.pipeline.debug VIDEO` runs only the measurement stages.

## Tests and synthetic evaluation

```bash
make test         # api: pytest (synth + claude_live markers excluded) · web: typecheck + lint + unit tests
make synth        # render 17 synthetic scenario videos + ground truth into data/synth (~35 s)
make eval         # run the pipeline on data/synth, write <name>.ir.json, check PLAN §11.4 thresholds
make calibrate    # Monte Carlo check that confidence bands are earned (high ≥ 90 % within target)
cd api && uv run pytest -m synth   # the same accuracy assertions as pytest
```

The synthetic suite covers card hover (with 30 fps / VFR / crf 28 / Retina / MOV / WebM /
no-cursor variants), button colour and press, dropdown, modal, accordion, stagger, and scroll and
static negatives. `make eval` prints `SUITE: PASS` only if every threshold holds: values,
start/duration, easing family ≥ 80 %, interaction type, relationships and confidence bands.

**Contract workflow.** `api/app/models/ir.py` and `api/app/api/schemas.py` are the source of
truth. After you change them, run `make contract`. Never edit `web/lib/types.ts` by hand. After
editing `docs/contract/sample-result.json`, run `pnpm -C web sync:fixture`.

## Make targets

| Target | What it does |
|---|---|
| `make help` | List targets |
| `make check-tools` | Check that uv, pnpm, ffmpeg, ffprobe are on `PATH` (and note if `claude` is missing) |
| `make setup` | `check-tools`, `uv sync` (api), `pnpm install` (web), create `data/jobs` |
| `make dev-api` / `make dev-web` | API on :8000 (with reload) / web on :3000 |
| `make test` (`test-api`, `test-web`) | Backend pytest + frontend typecheck, lint, unit tests |
| `make lint` | `ruff check` + `ruff format --check` + ESLint |
| `make contract` / `make contract-check` | Regenerate / verify `docs/contract/motion-spec.schema.json` + `web/lib/types.ts` |
| `make synth` / `make eval` / `make calibrate` | Synthetic videos / accuracy suite / confidence calibration |
| `make clean-jobs` | Delete all job data (uploads, artifacts, debug dumps, job DB) |

## Debugging (`MIMIC_DEBUG`)

Set `MIMIC_DEBUG=1` in `.env` and restart the API. Every job then also writes
`data/jobs/<id>/debug/`:

| File | Contents |
|---|---|
| `energy.csv` | For each pass-1 frame (30 fps): time, UI change energy, changed fraction, global shift + response, cursor position/state |
| `segments.json` | Active/stable segments, the primary interaction, windows, crop rectangle, warnings |
| `elements.json` | Element candidates: CSS boxes in state A/B, kind, parent, A→B warp |
| `masks/state_a.png`, `state_b.png`, `change_mask.png` | Median state frames (cropped) and the change mask |
| `tracks/<element>_<segment>_<property>.csv` | Per-frame series (time, value, quality) |
| `summary.json` | Elements, every series' endpoints, classifier features, heuristic interaction, target |

The dumps contain crops of the recording. They stay in `data/` (the API never serves them), and
`make clean-jobs` deletes them. `scripts/analyze.py --debug DIR` writes the same dumps for a
single file.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `missing required tool: ffmpeg` from `make setup`, `"ffmpeg": false` in `/api/health`, or a job fails with "ffprobe is not installed or not on PATH" | Install FFmpeg (`brew install ffmpeg`) or set `MIMIC_FFMPEG_BIN` / `MIMIC_FFPROBE_BIN`, then restart the API |
| `address already in use` on :8000 / :3000 | Another process holds the port: `lsof -iTCP:8000 -sTCP:LISTEN`, then stop it, or run `uvicorn ... --port <n>` and set `NEXT_PUBLIC_API_BASE_URL` to match |
| "Server unreachable" on the upload page | Start the API, check `NEXT_PUBLIC_API_BASE_URL`, and make sure `MIMIC_CORS_ORIGINS` includes the web origin |
| `too_long` / `too_short` | The clip is outside 0.5–15 s. Trim it to just the interaction, with about 1 s of stillness before it |
| `no_motion_detected` | No UI change was found. Make sure the animation is on screen and finishes within the clip |
| `no_stable_state` | The UI was already moving when recording started. Wait about 1 s before interacting |
| `unsupported_motion` | The whole page moved (scroll / page transition). Record a single component without scrolling |
| `decode_failed` | Re-export the clip as an H.264 MP4 |
| `interrupted` | The API restarted mid-job (often `--reload`). Upload again, or run uvicorn without `--reload` |
| Interpreter `unauthorized` | `claude_cli`: run `claude` on the API host to sign in. `openai_compat`: check `MIMIC_LLM_API_KEY` |
| Interpreter `unreachable` / `timeout` | Check that `MIMIC_LLM_BASE_URL` is up and reachable from the API host, or raise `MIMIC_LLM_TIMEOUT_S` |
| Interpreter `model_not_found` / `no_image_support` | Set `MIMIC_LLM_MODEL` to a vision-capable model that the gateway lists |
| Every distance is 2× too big or small | Wrong display scale. Re-upload with the right `1x` / `2x` / `3x` |

## Architecture

```
recording ─► Layer A: measurement (FFmpeg + OpenCV, deterministic) ─► every number
              probe → preview (background) → scan (30 fps) → decode windows (native fps)
              → elements → per-frame tracking → timing/easing fit + confidence → keyframes
          ─► Layer B: interpretation (optional AI) ─► labels, roles, hierarchy, type check
          ─► assemble IR (MotionSpec) ─► Technical / LLM prompt / JSON / CSS (pure templates)
```

- **Layer A owns the numbers.** The interpreter's output schema has no numeric fields except its
  own confidence, so it can't change a measured value.
- **The IR is the source of truth.** All four outputs are deterministic functions of it.
- Geometry is in CSS px: device pixels divided by the display scale.
- Jobs run in-process with one worker. Job state is in SQLite and files are under
  `data/jobs/<id>/`.

```
api/    FastAPI backend + measurement pipeline (uv, Python 3.13)
web/    Next.js 16 frontend (App Router, TypeScript, Tailwind, pnpm)
docs/   PLAN.md (plan + phase notes), contract/ (API + IR schema, shared fixture), screenshots/
data/   runtime data: jobs, job DB, synthetic videos (gitignored)
```

More detail: [`PRD.md`](PRD.md) (product), [`docs/PLAN.md`](docs/PLAN.md) (architecture,
algorithms, phase notes), [`docs/contract/`](docs/contract/) (JSON schema + sample result),
[`web/lib/README.md`](web/lib/README.md) (frontend data layer, mock mode).

## Known limitations

- **Display scale can't be read from pixels.** Auto is a guess. Set it explicitly and record at
  100 % browser zoom.
- Mimic has been validated only on **synthetic recordings** (clean encodes, ideal cursor). Real
  recordings will mostly get lower confidence. A re-encoded 15 s copy of a synthetic video moved
  one start time by about 23 ms.
- Easing: the family is usually right, but the exact bezier values are often low confidence.
  Motions shorter than about 150 ms are hard to fit. Strong ease-out curves are reported with
  "(duration uncertain)".
- Shadow and border-radius values are rough heuristics (the direction is reliable). Spring curves
  are flagged but not reconstructed. Rotation is detected but not measured.
- One interaction per recording. Extra interactions are ignored with a warning. Scrolling and
  page transitions are rejected (`unsupported_motion`). Drag, canvas and WebGL are out of scope.
- Long motions (> 150 frames in the analysed window) are analysed at a lower frame rate (warning
  `frames_subsampled`). Jobs run one at a time.
