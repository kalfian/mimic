# mimic

Upload a short screen recording of one UI interaction (hover, click, dropdown, modal, ...) and get
a structured motion specification back: measured values (OpenCV), optional AI labels, an
intermediate representation (IR) JSON, and generated Technical / LLM Prompt / JSON / CSS outputs.
All values are estimates from pixels, not source CSS.

Docs: [`PRD.md`](PRD.md) (product), [`docs/PLAN.md`](docs/PLAN.md) (plan + phase notes),
[`docs/contract/`](docs/contract/) (API + IR contract, shared fixture),
[`web/lib/README.md`](web/lib/README.md) (frontend data layer, mock mode).

## Quickstart

Needs [uv](https://docs.astral.sh/uv/), Node.js + [pnpm](https://pnpm.io/), and `ffmpeg` /
`ffprobe` on `PATH` (`brew install ffmpeg`). Optional: the `claude` CLI (logged in) for AI labeling.

```bash
make setup                                  # uv sync (api/, Python 3.13) + pnpm install (web/)
cp .env.example .env                        # optional: backend settings (MIMIC_*)
cp web/.env.local.example web/.env.local    # frontend: API URL / mock switch
make dev-api                                # FastAPI on http://localhost:8000 (--reload)
make dev-web                                # Next.js on http://localhost:3000
```

Open http://localhost:3000, drop a recording (MP4 / MOV / M4V / WebM, 0.5–15 s, ≤ 200 MB), pick
the display scale (Auto / 1x / 2x Retina / 3x) and, optionally, turn on AI labeling.

`make dev-api` reloads on every edit under `api/`, and a reload interrupts the running job (it
ends as `failed/interrupted`). For end-to-end sessions run `cd api && uv run uvicorn app.main:app
--port 8000` instead. `NEXT_PUBLIC_API_MOCK=1` in `web/.env.local` runs the frontend against the
fixture without a backend.

**Recording tips** (PRD §27): record at 100 % browser zoom and 60 fps if you can, keep the cursor
visible, start recording and wait about 1 s, perform **one** interaction, wait until the animation
has finished (return to the initial state for the reverse), then stop.

## How it works

```
recording ─► Layer A: measurement (OpenCV, deterministic) ─► every number
              probe → preview (background) → scan (30 fps) → decode windows (native fps)
              → elements → per-frame tracking → timing/easing fit + confidence → keyframes
          ─► Layer B: interpretation (optional AI) ─► labels, roles, hierarchy, type check
          ─► assemble IR (MotionSpec) ─► Technical / LLM prompt / JSON / CSS (pure templates)
```

- **Layer A owns the numbers.** The interpreter's output schema has no numeric fields except its
  own confidence; it can never change a measured value.
- **The IR is the source of truth** for all four outputs; they are deterministic functions of it.
- Every value carries a confidence (high ≥ 0.8, medium ≥ 0.5, low); uncertain changes (< 0.3) are
  listed separately. Geometry is in CSS px (the display scale divides device pixels).
- Jobs run in-process (one worker), state in SQLite, files under `data/jobs/<id>/`.

## AI labeling (interpreter modes)

AI labeling is **opt-in per upload (default off)**: without the toggle no keyframe leaves the
machine. It can also be added later to a finished job ("re-run", below). The backend is
configured server-side only, in `.env`:

| `MIMIC_INTERPRETER` | What runs when a job opts in | Settings |
|---|---|---|
| `claude_cli` (default) | `claude -p` subprocess with its own login under `$HOME` (no API key) | `MIMIC_CLAUDE_BIN`, `MIMIC_CLAUDE_MODEL` (`sonnet`), `MIMIC_CLAUDE_TIMEOUT_S` (120) |
| `openai_compat` | any OpenAI-compatible gateway (e.g. 9Router); keyframes go as images | `MIMIC_LLM_BASE_URL`, `MIMIC_LLM_API_KEY`, `MIMIC_LLM_MODEL`, `MIMIC_LLM_TIMEOUT_S`, `MIMIC_LLM_SUPPORTS_JSON_SCHEMA` (`auto`) |
| `none` | nothing; heuristic labels only | — |

Example for a local gateway such as 9Router (values are placeholders):

```bash
MIMIC_INTERPRETER=openai_compat
MIMIC_LLM_BASE_URL=http://localhost:20128/v1
MIMIC_LLM_API_KEY=<your-gateway-token>
MIMIC_LLM_MODEL=<model-id-that-accepts-images>
```

Tokens belong only in your local `.env` (gitignored). They are never sent to the browser, logged,
or written to job files. `GET /api/health` reports whether the mode is configured without calling
it; "Test connection" on the upload page (`POST /api/interpreter/check`) does a real round trip.
Any interpreter failure (timeout, error, bad output) falls back to heuristic labels and the result
says so; a job never fails because of it.

**Re-run interpretation.** `POST /api/jobs/{id}/interpret` with `{"use_interpreter": true}`
re-labels a succeeded job from its stored measurements (`measurement.json`) without decoding the
video again. The job goes `queued → interpreting → generating → succeeded`; `409 already_running`
while it is busy, `409 not_ready` for failed jobs. This is the explicit user action that sends the
keyframes; `false` goes back to heuristic labels.

## Debugging (`MIMIC_DEBUG`)

`MIMIC_DEBUG=1` (in `.env`, restart the API) makes every job write `data/jobs/<id>/debug/`:

| File | Contents |
|---|---|
| `energy.csv` | per pass-1 frame (30 fps): time, UI change energy, changed fraction, global shift + response, cursor position/state |
| `segments.json` | active/stable segments, the primary interaction, windows, crop rectangle, warnings |
| `elements.json` | element candidates: CSS boxes in state A/B, kind, parent, A→B warp |
| `masks/state_a.png`, `state_b.png`, `change_mask.png` | median state frames (crop) and the change mask |
| `tracks/<element>_<segment>_<property>.csv` | per-frame series (time, value, quality) |
| `summary.json` | elements, every series' endpoints, classifier features, heuristic interaction, target |

The dumps contain crops of the recording; they stay in `data/` (never served by the API) and are
deleted by `make clean-jobs`. `scripts/analyze.py --debug DIR` writes the same dumps for one file,
and `uv run python -m app.pipeline.debug VIDEO` runs the measurement stages only.

## Make targets

```bash
make test        # backend pytest + frontend typecheck/lint/unit tests
make lint        # ruff + eslint
make contract    # regenerate docs/contract/motion-spec.schema.json + web/lib/types.ts
make contract-check   # fail if those generated files are stale
make synth       # render the synthetic test videos into data/synth (~35 s)
make eval        # run the pipeline on data/synth, check the PLAN §11.4 accuracy thresholds
make calibrate   # Monte Carlo check that confidence bands are earned (high ≥ 90 % in target)
make clean-jobs  # delete all job data (uploads, artifacts, debug dumps, job DB)
```

Analyze one file without the server:

```bash
cd api
uv run python scripts/analyze.py rec.mov --pixel-ratio 2                      # summary + timings
uv run python scripts/analyze.py rec.mp4 --print llm_prompt                   # one output
uv run python scripts/analyze.py rec.mp4 --out result.json --keyframes kf/ --debug dbg/
```

**Synthetic evaluation.** `make synth` renders 17 scenario videos (card hover incl. 30 fps / VFR /
crf 28 / Retina / MOV / WebM / no-cursor variants, button colour and press, dropdown, modal,
accordion, stagger, plus scroll and static negatives) with exact ground truth. `make eval`
compares the pipeline against it and prints `SUITE: PASS` only if every §11.4 threshold holds
(values, start/duration, easing family ≥ 80 %, interaction type, relationships, confidence-band
calibration). `cd api && uv run pytest -m synth` asserts the same.

**Contract workflow.** The Pydantic models in `api/app/models/ir.py` and `api/app/api/schemas.py`
are the source of truth. After changing them run `make contract` (never edit `web/lib/types.ts`);
after editing the fixture run `pnpm -C web sync:fixture`.

## Known limitations

- **Display scale** cannot be read from pixels. `Auto` guesses 2x when the recording is ≥ 2560 px
  wide or ≥ 1600 px tall; set it explicitly and record at 100 % browser zoom, or px values are off
  by that factor.
- Validated on synthetic recordings only (clean encodes, ideal cursor). Real recordings mostly
  lower confidence; a re-encoded 15 s copy of a synthetic video moved one start time by ~23 ms.
- Easing: the family is usually right, exact bezier values often low confidence; motions under
  ~150 ms are barely fittable. Strong ease-out curves report "(duration uncertain)".
- Shadow and border-radius are heuristics (direction is reliable, values rough). Spring curves are
  flagged, not reconstructed. Rotation is detected, not measured.
- One interaction per recording (extra ones are ignored with a warning). Scroll and page
  transitions are rejected (`unsupported_motion`); drag, canvas/WebGL are out of scope.
- Long motions (> 150 frames in the analysed window) are analysed at a reduced frame rate
  (warning `frames_subsampled`). Jobs run one at a time.

## Layout

```
api/   FastAPI backend + measurement pipeline (uv, Python 3.13)
web/   Next.js frontend (App Router, TypeScript, Tailwind, pnpm)
docs/  plan + contract
data/  runtime data (gitignored)
```
