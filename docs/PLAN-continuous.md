# Mimic — Continuous / looping motion + drag (scroller) analysis

Status: **approved** (2026-10-03); **P0 done**, P1 tracks done, **P2 done** (see "P0 notes",
"P2 notes" … "P2e notes" at the end; §14 acceptance iteration 1 in P2e). Base: `master` @ fa68247.
Scope owner decision: this deliberately extends PRD §26 ("complex drag-and-drop", "precise spring
parameter reconstruction" stay out). In scope is **one single-axis scroller** (marquee / auto-scrolling
carousel) that may be **hovered/pressed, dragged, released (inertia/snap) and resume autoplay**.
Everything in `docs/PLAN.md` stays authoritative for the transition pipeline; nothing there is
reopened except the additive contract changes listed in §5.

---

## 0. Context

### 0.1 Motivating failure (numbers only; frames were not viewed)
macOS `.mov`, 904×786, VFR ≈55.7 fps, 9.3 s, pixel ratio auto=1 → `no_motion_detected` in scan.

- A horizontal band (CSS ≈ x 89, y 301, 746×194, ~16 % of the frame) changes from t=0
  (≈30k px² changed per frame even before any interaction).
- `scan.energy_thresholds` takes median+MAD of the **whole** series → floor ≈93.8k,
  `τ_hi` ≈280k > max energy ≈125k → zero runs → `no_motion_detected` (misleading).
- Cursor: 0 visible frames. Per-pixel change is intermittent (slow content, edges only), so a
  per-pixel "always changing" test would also miss it.
- Band velocity (px/s, +x = content moves right): 0–1.5 s +38–40 (linear autoplay);
  1.5–2.0 s 39→27→15 (decelerate); 2.0–2.5 s −1200…−1500 (drag); 2.75–3.5 s −494→−150→−41→−20
  (inertia, τ≈0.2 s); 3.75–5.25 s ≈−800 (drag) then decay; 5.5–6.25 s +600…+990 (drag right) →
  ≈+70 by 6.5 s; 7.0–7.5 s ≈−1700 then **abrupt** stop at 7.75 (snap, or the pointer stopped
  before release); 7.75–8.5 s 0→17→24→38 (resume ramp ≈0.75 s). vy ≈ 0. phaseCorrelate
  response 0.55–0.7 during fast drags.

### 0.2 Codebase facts that shape the design
- `scan.py` `build_scan_result` is a single linear path: cursor → energy → `segment_energy`
  (global median+MAD) → `no_motion_detected` / `unsupported_motion` / `no_stable_state` →
  pairing. Pass 1 keeps only blobs (CSS rects), changed fraction, a full-frame phaseCorrelate
  when > 30 % changed, and 240-wide thumbnails. **No per-pixel/cell history is kept.**
- Pass 2 (`decode.decode_windows`) is window-based, capped at 150 frames per window, and keeps
  crops in memory. Continuous analysis needs the **whole video** for one region → needs a
  streaming region reader.
- `WindowFrames.cursor_masks` are "exclude these pixels" masks consumed by every A4/A5 stage —
  reusable for masking ambient regions in the transition path without touching A4/A5 code.
- IR (`models/ir.py`) is strict (`extra="forbid"`, closed `Literal` sets, `segments` min 1,
  `direction` ↔ segment-set validator, `schema_version: Literal["0.1"]`). `web/lib/types.ts` is
  generated (`make contract`); `format.ts`/`errors.ts` use `Record<Enum, …>` so new enum
  members fail typecheck until handled (good: forces UI coverage).
- Generators are pure `render(spec) -> str`, dispatched by `render_all` (4 fixed outputs);
  `tests/unit/test_generators.py` enforces **number provenance** (every printed number maps to an
  IR value after rounding; CSS may also use `0`/`1`).
- Synth suite: tweens on a `Timeline` (piecewise, non-overlapping) — cannot express
  velocity-integrated motion, exponential decay or wrap-around. Truth is derived from the same
  scene definition (no drift). `make eval` = 17/17 today.
- `measurement.json` (persist.py) is an allow-listed tagged JSON of `Measurement` with
  `FORMAT_VERSION = 1`; re-run interpretation depends on it.
- Concurrent change (in progress, other coder): the existing LLM prompt (`generate/llm_prompt.py`)
  now asks the coding LLM for **one self-contained, responsive HTML file** (HTML + CSS + inline
  JS, no external deps/build step, neutral placeholder content). This plan does **not** change
  that file's existing behaviour; the continuous prompt adopts the same output contract (§6.2).

## 1. Prior decisions that constrain this plan
- Layer A numbers are the only source of numbers; Layer B (interpreter) never produces numbers
  (PLAN §7). Applies unchanged: interpreter only labels the scroller.
- §11.4 thresholds are never relaxed to make a run pass (Phase 4 note). New targets in §8.4 are
  set up-front with justification.
- Real recordings are diagnosed numerically; frames are not viewed without consent (vault MOC).
- AI labeling stays opt-in/default off; keyframes may be sensitive.
- No pickle for persisted measurements (allow-listed JSON).

---

## 2. Approach (summary)

1. **Regime detection before segmentation.** Pass 1 additionally keeps a coarse per-frame
   **cell-activity grid**. Regions that are active from the **start** of the recording
   (leading-window activity, cursor removed) are **ambient**. No ambient region → the current
   path runs **unchanged** (byte-identical IR for all existing scenarios). Ambient region(s)
   present → analyse them as continuous motion and/or mask them out of the transition path.
2. **Continuous-scroller measurement** on a streaming full-rate crop of the ambient region:
   coherent single-axis translation test, per-frame displacement (prediction-guided NCC +
   ECC translation refine, blur-matched retry), VFR-aware timestamps, panorama for loop period /
   card pitch.
3. **Phase segmentation** of the displacement/velocity profile with a small grammar
   (autoplay → decelerate/paused → drag → inertia|snap|stop → paused → resume → autoplay),
   model fits **in the position domain** (constant, exponential decay, eased velocity ramp,
   eased snap tween), cursor used when visible.
4. **IR 0.2**: root `mode: "transition" | "continuous"` + optional `continuous` section
   (phases + aggregated behaviour, all values with confidence). Transition specs are unchanged
   apart from `mode` and `schema_version`.
5. **Generators**: continuous renderers in a new `generate/continuous/` package dispatched by
   `render_all`; new optional `js` output (driver snippet). The LLM prompt follows the
   single-responsive-HTML-file contract.
6. **UI**: velocity-over-time chart with phase bands replaces the transition timeline in
   continuous mode; behaviour summary replaces per-element rows.
7. **Synth**: kinematic drivers (velocity-integrated, wrap-around) + 11 new scenarios + ambient
   regression scenario + negative.

### Rejected alternatives (do not re-propose)
- **Leading-window baseline for the global energy threshold only** (replace median by the
  first-0.5 s median). Fixes the threshold but then the scroller is one 9 s "active run" fed to
  the A/B-state pipeline, which produces a meaningless translateX transition. Also changes
  thresholds for existing scenarios → risks the 17/17 regression.
- **Per-pixel persistence map** ("pixel changes in ≥ 60 % of frames"). Fails on the real case:
  slow textured content changes only at edges, intermittently. Cell-level (16 CSS px) activity
  with lagged diff is robust.
- **Encode phases as IR `transitions`** (e.g. inertia as a translateX ease-out transition).
  Segment ids are a closed set tied to forward/reverse semantics; autoplay is unbounded; inertia
  is velocity-parametrised (τ), not endpoint-parametrised. It would silently mislead every
  existing consumer.
- **Separate `ContinuousSpec` root type / second endpoint.** Doubles the contract surface
  (`ResultEnvelope`, re-run, persistence, web types). A `mode` discriminant on one root keeps one
  envelope and one `types.ts`.
- **Velocity-domain fitting** (differentiate then fit). Differentiation amplifies tracking noise
  and VFR jitter; position-domain fits use the integrated models directly.
- **Frame-count friction constants** (e.g. "×0.92 per frame") in outputs. Frame-rate dependent
  and not an IR value; outputs use `τ` with `v·exp(−dt/τ)` instead.
- **Treating the abrupt stop as snap by default.** Without a cursor, "pointer held still then
  released" and "hard snap" look identical; snap requires grid evidence (§4.4).

---

## 3. Regime detection (Track R)

### 3.1 Pass-1 activity grid (`scan.Pass1Accumulator`)
- Cell size `cell_css = 16` CSS px. Per frame, from the **existing** thresholded+opened diff
  `d` (before dilation) and the existing lagged diff (`SLOW_LAG`): cell active if ≥ 0.5 % of its
  pixels changed in either diff (`cv2.resize(d, grid, INTER_AREA) > 0.005`).
- Store `activity: np.ndarray[bool] (N, gh, gw)` in `Pass1Stats` (1280×800 CSS → 80×50 cells;
  900 frames → 3.6 MB). No change to existing per-frame outputs.

### 3.2 Ambient regions (`pipeline/continuous/regime.py`, pure)
1. Remove cursor: for each frame, clear cells under `cursor.boxes_css[i]` (cursor track from the
   unchanged `build_cursor_track`).
2. Leading window `[t_first, t_first + lead_window_s (0.6 s)]`, ≥ 10 pass-1 frames.
   `lead_frac[cell]` = fraction of leading frames where the cell is active.
3. Ambient cells: `lead_frac ≥ 0.5`. Morph close 3×3 cells, connected components, keep
   components ≥ 2 cells. Up to `max_ambient_regions = 3` by area.
4. **Only motion present from the start counts as ambient.** Motion that starts after a static
   lead is regime A by definition (this keeps every existing scenario and every
   "long animation after hover" case on the current path).
5. Output `RegimeReport{ambient: list[AmbientRegion{rect_css, cells, lead_frac}], cursor}`.

### 3.3 Decision (`regime.decide_mode`, pure; inputs from §3.2, §4 and the masked scan)
```
no ambient regions                          -> TRANSITION (current path, untouched)
ambient regions present:
  masked = build_scan_result(stats, ..., ambient=regions)   # may raise; caught
  scrollers = [r for r in analysed regions if r.is_scroller]
  page-scroll check: a scroller covering > 60 % of frame area, or ≥ 90 % of both
                     frame dimensions                        -> error unsupported_motion
  scroller with interaction phases (any of decelerate/paused/drag/inertia/snap/resume)
                                             -> CONTINUOUS (largest such scroller)
  scroller autoplay-only AND masked has runs -> TRANSITION (masked) + warning ambient_motion_masked
  scroller autoplay-only, masked has no runs -> CONTINUOUS (marquee)
  no scroller AND masked has runs            -> TRANSITION (masked) + warning ambient_motion_masked
  no scroller AND no runs                    -> error continuous_motion_unsupported (region named)
```
- Extra scrollers beyond the chosen one → warning `extra_scrollers_ignored`.
- Discrete runs outside the scroller in CONTINUOUS mode → existing `extra_segments_ignored`.

### 3.4 Masked transition path
- `build_scan_result(stats, scale, params, ambient=None)`: when `ambient is None` the function
  body is exactly today's. When set: per-frame blob area is reduced by its overlap fraction with
  the (8 px dilated) ambient rects, slow-blob likewise; ROI unions skip the masked parts; the
  global-motion check ignores masked area. Thresholds then work because the remaining energy has
  a static baseline.
- Pass 2: `regime.apply_ambient_mask(wfs, regions)` ORs the ambient rects into
  `WindowFrames.cursor_masks` (called in `run.measure`, integration). A4/A5 already honour
  these masks — no A4/A5 edits.

### 3.5 Errors / warnings (closed sets, contract change in P0)
- New error `continuous_motion_unsupported` (422). Message is specific, e.g. "Part of the page
  (around x 89, y 301, 746×194 px) moves continuously from the start, but not as a single
  horizontal or vertical scroller, so it can't be measured. Record a component that rests
  before you interact, or pause that animation." Variants: two-axis/rotating content;
  untrackable (coherence < threshold); several opposing scrollers in one region that could not
  be split.
- `no_motion_detected` is raised **only** when there are no ambient regions and no runs (i.e.
  never for the motivating video).
- New warnings: `ambient_motion_masked` (info; names region and, if a scroller, its speed),
  `extra_scrollers_ignored` (info), `tracking_degraded` (warn; % of frames below quality and the
  phases affected), `loop_period_not_observed` (info; content never repeated within the
  recording).

---

## 4. Continuous-scroller measurement (Track C)

All new modules under `api/app/pipeline/continuous/`. Internal times are float seconds,
geometry CSS px, like the rest of Layer A.

### 4.1 Streaming region reader (`decode.iter_region`)
- `iter_region(path, probe, scale, crop_css, params, max_fps=60) -> Iterator[(t, gray_f32)]`:
  ffmpeg `-fps_mode passthrough`, crop + scale to analysis px, gray; `t` from `probe.frame_ts`
  (same consistency check / `timestamps_estimated` fallback as `decode_windows`); > 60 fps keeps
  every k-th frame. Streams; never holds the whole video.
- Duplicate frames (max-abs-diff ≤ 1 vs previous) are **skipped** (not zero-velocity samples);
  the next real frame's displacement spans the merged `dt`.
- Crop = ambient rect grown by 2 cells, clamped. The scroller viewport box is refined from the
  union of changed pixels across the stream (reported `region`, confidence ≤ 0.7).

### 4.2 Coherence / axis test (`displacement.py`)
- Frame-to-frame translation by `cv2.phaseCorrelate` (Hann window) on the region crop for the
  first ≤ 1 s (or the whole stream when the region is small).
- Scroller iff ≥ 70 % of frames have response ≥ 0.3, and median |d_perp| < 0.25 px with axis
  dominance |d_axis| / |d_perp| ≥ 5 over frames with |d| ≥ 0.3 px. Axis = x or y.
- **Opposing sub-scrollers** (two marquee rows in one ambient component): split the region into
  strips perpendicular to the axis (one cell high), estimate each strip's shift over the
  leading window, cluster strips by velocity sign/magnitude (|Δv| > max(10 px/s, 25 %)), re-test
  each cluster. Largest coherent cluster wins; others → `extra_scrollers_ignored`.

### 4.3 Displacement time series (`displacement.py`)
Per real frame i (after the coherence test passes):
1. **Prediction**: `d̂_i = v̂_{i−1}·dt_i` (constant velocity; clamp |acceleration| to
   `max_accel_px_s2` (60 000) when choosing the search window).
2. **Initial**: phaseCorrelate on the Hann-windowed crops; if response < 0.3 or
   |d − d̂| > max(8 px, 0.5·|d̂|), fall back to the prediction.
3. **Refine**: 1-D NCC (`TM_CCOEFF_NORMED`) of the previous crop's inner part (inset by
   `|d̂| + 16 px` along the axis) against the current crop inside a window of ±max(6 px, 0.3·|d̂|)
   around the initial estimate; parabolic sub-pixel; then `findTransformECC(MOTION_TRANSLATION)`
   warm-started (gaussFiltSize 3). Quality `q_i` = NCC peak (ECC rho if it succeeded).
4. **Blur-matched retry** if `q_i < 0.6`: blur the reference along the axis with a box kernel of
   length {⌈|d|/2⌉, ⌈|d|⌉} px and repeat step 3; keep the best. (Covers recorder frame blending
   and compression smear at > 1500 px/s.)
5. **Multi-scale** if |d̂| > 0.35·crop length along the axis: run steps 2–3 at ½ resolution first.
6. **Consistency**: every 4th frame also estimate `d(i, i−2)`; if it disagrees with
   `d_i + d_{i−1}` by > 1 px, mark both samples low quality (q ≤ 0.4).
7. Positions `x_i = Σ d` (CSS px, signed; +x = content moves right / +y = down), `t_i` from
   timestamps. Output `DisplacementSeries{times, pos, quality, axis, region, dt_median}`.
- Smoothed velocity for labelling/UI only: local linear regression on `pos` over a ±2-sample
  window (non-uniform `t`). **Fits use `pos`, never the derivative.**
- Frames with `q < 0.5` > 10 % inside any phase → warning `tracking_degraded` and that phase's
  value confidences × 0.7.
- Cursor pixels: the crop's cursor mask (pass-1 boxes) is applied as the ECC mask; NCC ignores
  it (a pointer is < 1 % of a typical scroller crop).

### 4.4 Panorama: loop period and card pitch (`panorama.py`)
- Paste each frame's crop into a 1-D-extended canvas at `round(x_i)` (along the axis), running
  median per column (≤ 4 contributions kept per column, memory ≈ crop height × (travel + crop)).
- **Loop period P**: smallest lag L ≥ 0.25·crop length with NCC(pano, pano shifted by L)
  ≥ 0.9 over an overlap ≥ 0.5·crop length. Needs total travel ≥ L + 0.5·crop length; otherwise
  `loop.observed = false` + warning `loop_period_not_observed` (never guessed).
- **Pitch p** (card spacing): dominant period of the perpendicular-edge energy profile (vertical
  gap edges for a horizontal scroller) via autocorrelation, first peak with prominence ≥ 0.3;
  must be < P when P is observed. Confidence ≤ 0.8. Used for snap detection and STRUCTURE text.

### 4.5 Phase segmentation (`phases.py`, `kinematics.py`)
Grammar (allowed transitions; anything else is relabelled or merged):
```
autoplay -> decelerate -> paused -> drag
autoplay -> drag                         (press stops autoplay instantly)
drag -> drag                             (direction reversal kept inside one drag, as a sub-event)
drag -> inertia | snap | stop            (release)
inertia -> drag                          (re-grab during momentum)
inertia -> snap -> paused
inertia | snap | stop -> paused -> resume -> autoplay
decelerate -> resume                     (hover left without dragging)
paused -> resume
```
Kinds: `autoplay, decelerate, paused, drag, inertia, snap, stop, resume, unknown`.

Steps:
1. **Autoplay speed** `v_auto`: robust estimate over the longest window(s) where the smoothed
   velocity has low local variance (cv < 0.15) and |v| < 400 px/s, preferring the leading window.
   If the leading window is not autoplay (drag from t=0) → `autoplay = null`.
2. **Coarse labels** per sample: `A` if |v − v_auto| ≤ max(3 px/s, 0.12·|v_auto|);
   `Z` if |v| ≤ max(5 px/s, 0.15·|v_auto|); else `M`. Runs shorter than 3 samples are absorbed.
3. **Split each `M` run** into `drag` + release phase: for each candidate release index r in the
   run (last 70 % of the run), fit the tail `[r, end]` (position domain) with
   (a) exponential `x(t) = x_r + v_∞(t−t_r) + (v_0 − v_∞)·τ·(1 − e^{−(t−t_r)/τ})`, `v_∞ ∈ {0, v_auto}`;
   (b) eased snap tween `x(t) = x_r + D·E((t−t_r)/T)` with E ∈ {ease-out family, Appendix B
   curves}; (c) none (stop). Cost = tail SSE/σ² + BIC penalty; head `[start, r)` is drag
   (unmodelled, cost = fixed per-sample noise floor so the split prefers a parametric tail only
   when it explains the data). Pick the minimum-cost r and model.
   - Require exponential: ≥ 6 samples, τ ∈ [40 ms, 2 s], |v_0| ≥ 3·zero-tolerance, monotone
     |v| decrease (≤ 1 violation).
4. **Stop vs snap** (abrupt end of a drag/inertia): |v| falls from > 300 px/s to ≈0 within
   ≤ 2 frames. Label `snap` only if the rest position is on the pitch grid (below); else `stop`
   with note "pointer likely stopped before release, or hard snap (ambiguous without cursor)".
5. **Snap evidence**: rest positions `x_rest_k` (end of each inertia/snap/stop with v≈0 for
   ≥ 100 ms) → phase φ_k = (x_rest_k − x_ref) mod p. Snap if ≥ 2 rests and circular resultant
   R ≥ 0.9, or 1 rest with a fitted snap tween ending within 2 px of a grid line (low conf ≤ 0.5).
   Overshoot: position passes the rest value by > 1 px then returns within the phase → flag
   `overshoot` ("spring-like; spring parameters not reconstructed").
6. **Decelerate / resume**: `A → Z` and `Z → A` transitions fitted as eased velocity ramps in the
   position domain (`x(t) = x_0 + ∫ v(t) dt`, `v = v_a + (v_b − v_a)·E((t − t0)/D)`), reusing
   `easing.py` candidates and the joint t0/D grid idea of PLAN §6.7 (same Bezier evaluator).
   A decelerate that is cut by a drag before reaching `Z` is kept with `interrupted: true`.
7. **Resume delay** = resume `t0` − end of the preceding rest-reaching phase
   (`delay_after_rest_ms`) and − release time (`delay_after_release_ms`). `direction_preserved`
   = sign(v after resume) == sign(v_auto before).
8. **Cursor fusion (when `cursor.visible`)**:
   - Cursor enters the region within [−400, +100] ms of a decelerate t0 → `pause.on = hover`
     (confidence 0.85); cursor stationary inside the region at decel t0 → `press` (0.7);
     no cursor → `unknown` (behaviour still reported, confidence of the trigger ≤ 0.4).
   - During drag, cursor velocity / content velocity ≈ 1 (±0.15) over ≥ 60 % of drag samples →
     `drag.follows_pointer = true` with measured ratio; release = first sample where content and
     cursor velocities diverge (> 25 %), overrides step 3's r if within ±3 frames.
   - In-band cursor detection (motion-compensated residual) is **P4**, not P1 (§9).
9. **Aggregation into behaviour** (`kinematics.aggregate`): per behaviour, weighted median over
   phase instances (weights = per-phase confidence), `instances` count, `spread` (IQR). Agreement
   across ≥ 2 instances raises confidence (factor `clip(1 − spread/median, 0.6, 1)` replaces 1).

### 4.6 Confidence (`continuous/confidence.py`; calibrate in P2 like `make calibrate`)
- `q_track` = clip((mean q in phase − 0.6)/0.35, 0, 1).
- Speed: `q_track · clip(T_obs / 1 s, 0, 1) · clip(1 − cv/0.1, 0, 1)`, cap 0.95.
- τ: from the fit covariance `rel_se(τ)` → `q_track · clip(1 − 2·rel_se, 0, 1) ·
  clip((n − 4)/12, 0, 1)`, cap 0.9.
- Release velocity: `q_track · clip(1 − dt/τ, 0, 1)`, cap 0.85.
- Phase label: model-separation factor `clip(0.4 + ΔBIC/10, 0, 1)` (like `sep_factor`).
- Durations / boundaries: `clip(1 − dt/D, 0, 1) · fit factor`, + existing `is_vfr` −0.05 and
  `timestamps_estimated` −0.15 penalties.
- Caps: snap step 0.8, snap (single rest) 0.5, pause trigger without cursor 0.4, loop period 0.9,
  region box 0.7, pitch 0.8. Bands use the existing `band_for`.

### 4.7 Orchestrator (`continuous/measure.py`) and assemble (`continuous/assemble.py`)
- `analyse_ambient(video, info, scale, regime, params) -> list[ScrollerAnalysis]` (coherence,
  displacement, panorama, phases per candidate region; no IR).
- `assemble_continuous(m: Measurement, interp, ...) -> MotionSpec` builds the IR of §5 (one
  `MotionElement` kind `scroller`, `continuous` section, samples decimated to ≤ 900 points).
- `interpretation_input_continuous(...)`: element summary "scroller; content moves {axis} at
  about N px/s; K drags" + phase keyframes; the interpreter labels the scroller and writes
  `structure` only. **Interaction type in continuous mode is always heuristic** (Layer B cannot
  see velocities); `merge_type` is bypassed.

---

## 5. IR / contract extension (P0)

`schema_version` → `"0.2"` (type `Literal["0.1", "0.2"]` so stored 0.1 results still load for
re-run; the pipeline always emits 0.2). All additions are optional/defaulted; a transition spec
differs from today only in `schema_version` and `mode: "transition"`.

### 5.1 Closed vocabularies (additions)
| Literal | Added members |
|---|---|
| `InteractionType` | `continuous` (autoplay only), `drag` (scroller with pointer drag) |
| `Pattern` | `marquee`, `carousel` |
| `TriggerKind` | `autoplay`, `drag` |
| `Direction` | `continuous` (↔ `segments == []`) |
| `ElementKind` | `scroller` |
| `Role` | `scroller` |
| `WarningCode` | `ambient_motion_masked`, `extra_scrollers_ignored`, `tracking_degraded`, `loop_period_not_observed` |
| `ErrorCode` | `continuous_motion_unsupported` (422) |
| `KeyframeKind` (schemas) | `phase`; `KEYFRAME_NAME_RE` adds `phase_[1-9][0-9]?` |

### 5.2 New models (`models/ir.py`)
```
MotionSpec
  mode: "transition" | "continuous" = "transition"
  continuous: ContinuousMotion | null = null
  segments: list[Segment]          # min_length 0; validator: transition mode needs >= 1

ContinuousMotion
  element_id: ElementId            # MotionElement with kind "scroller"
  axis: "x" | "y"
  region: Box                      # viewport, CSS px
  region_confidence: Confidence
  sign_convention: "positive = content moves right (x) / down (y)"   # Literal, documents sign
  autoplay: Autoplay | null
  pitch_px: MeasuredNumber | null  # card spacing if observed
  phases: list[Phase]              # chronological, non-overlapping
  behavior: Behavior
  span_ms: { start_ms: Ms, end_ms: Ms }   # analysed span
  samples: list[tuple[int, float, float, float]]   # [t_ms, v_px_s, pos_px, quality], <= 900,
                                                   # excluded from JSON export (like transition samples)

Autoplay
  direction: "left" | "right" | "up" | "down"
  speed_px_s: MeasuredNumber        # |v|, CSS px/s
  velocity_px_s: float              # signed (same number, sign = direction)
  easing: Literal["linear"]
  loop: { observed: bool, period_px: MeasuredNumber | null, duration_ms: MeasuredNumber | null }
                                    # duration_ms = period / speed (stored so outputs can quote it)

Phase
  id: "p1".. ; kind: autoplay|decelerate|paused|drag|inertia|snap|stop|resume|unknown
  start_ms, end_ms: Ms
  v_start_px_s, v_end_px_s, v_peak_px_s: float (signed)
  displacement_px: float (signed)
  fit: PhaseFit | null
  interrupted: bool = false         # e.g. decelerate cut by a drag, inertia re-grabbed
  evidence: "velocity" | "cursor" | "velocity+cursor"
  confidence: Confidence            # label confidence
  notes: list[str]

PhaseFit (discriminated on model)
  constant:    { velocity_px_s: MeasuredNumber }
  exponential: { tau_ms: MeasuredNumber, v0_px_s: MeasuredNumber, v_inf_px_s: float }
  ramp:        { from_px_s, to_px_s: float, duration_ms: MeasuredNumber, easing: Easing }
  tween:       { distance_px: MeasuredNumber, duration_ms: MeasuredNumber, easing: Easing }   # snap / tween-style inertia

Behavior   (what generators print; each value aggregated over phase instances)
  pause:   { on: "hover"|"press"|"unknown", on_confidence: Confidence,
             decel_ms: MeasuredNumber | null, easing: Easing | null, stops_completely: bool } | null
  drag:    { count: int, follows_pointer: bool | null, pointer_ratio: MeasuredNumber | null,
             peak_speed_px_s: MeasuredNumber } | null
  inertia: { model: "exponential" | "tween", tau_ms: MeasuredNumber | null,
             duration_ms: MeasuredNumber | null, easing: Easing | null,
             release_speed_min_px_s: float, release_speed_max_px_s: float, instances: int } | null
  snap:    { kind: "grid" | "abrupt_ambiguous", step_px: MeasuredNumber | null,
             duration_ms: MeasuredNumber | null, easing: Easing | null, overshoot: bool } | null
  resume:  { delay_after_rest_ms: MeasuredNumber, delay_after_release_ms: MeasuredNumber | null,
             ramp_ms: MeasuredNumber, easing: Easing | null, to_speed_px_s: float,
             direction_preserved: bool } | null
```
Validators (all in `ir.py`): `mode == "continuous"` ⇔ `continuous is not None` ⇔
`interaction.direction == "continuous"` ⇔ `interaction.type in {continuous, drag}`; continuous mode
requires `segments == transitions == relationships == []`; `continuous.element_id` exists with
kind `scroller`; phases sorted, ids unique `p1..`, `start < end`, fit model allowed for kind
(autoplay→constant, inertia→exponential|tween, snap→tween|null, decelerate/resume→ramp,
drag/stop/paused/unknown→null); `autoplay.velocity_px_s` sign matches `direction` and |value| ==
`speed_px_s.value` (1e-6); `loop.observed` ⇔ `period_px is not None`. `interaction.total_duration_ms.forward`
= analysed span (documented), `reverse = null`. `target_element_id` = scroller.

### 5.3 Other contract files
- `generate/__init__.py`: `Outputs.js: str | None = None` (serialized `js`); `render_all`
  dispatches by `spec.mode` to `GENERATOR_MODULES` (unchanged) or `CONTINUOUS_GENERATOR_MODULES`
  (`generate/continuous/*`, + `js`). Transition mode → `js = null`.
- `export_json.py`: also exclude `continuous.samples`.
- `contract.py`: export `PHASE_KINDS`, `CONTINUOUS_SAMPLE_MAX` to `types.ts`; regenerate
  `docs/contract/motion-spec.schema.json` and `web/lib/types.ts`.
- New fixture `docs/contract/sample-continuous-result.json` (hand-authored, synthetic numbers
  shaped like §0.1, no real content), validated by `test_ir_contract.py`.
- `models/measure.py`: `AmbientRegion`, `RegimeReport`, `DisplacementSeries`, `PhaseCandidate`,
  `ScrollerAnalysis`, `ContinuousMeasurement`; `Measurement.continuous: ContinuousMeasurement |
  None = None`. `persist.py`: allow-list the new dataclasses; keep `FORMAT_VERSION = 1` if the
  decoder accepts the missing field via its default (verify with a v1 fixture), else bump to 2
  (old jobs then answer `not_ready` on re-run, already handled).
- `params.py`: new frozen group `ContinuousParams` holding every threshold named in §3–§4.

---

## 6. Generators (Track G) — `api/app/generate/continuous/`

All pure, deterministic, provenance-checked (§6.5). Hedging and rounding reuse `phrasing.py`
(px, ms → 10 ms, confidence wording, uncertain < 0.3 → "Uncertain observations"). Speeds:
≥ 100 px/s round to 10, else to 1, printed "≈N px/s".

### 6.1 `technical.py`
`CONTINUOUS MOTION` (scroller label, axis, region size, pitch if observed) → `AUTOPLAY`
(direction, speed, linear, loop period/duration or "loop length not observed") → `PHASES`
(table: #, kind, start–end ms, v start→end, key fit value, confidence) → `BEHAVIOUR` (pause,
drag, inertia, snap, resume blocks with confidence) → `UNCERTAIN OBSERVATIONS` → `NOTES`
(warnings, sign convention, "estimated values").

### 6.2 `llm_prompt.py` (continuous) — follows the single-HTML-file contract
- **Output contract (same as the transition prompt, being added concurrently):** ask for ONE
  self-contained, responsive HTML file (HTML + CSS + inline JS, no external dependencies or build
  step, neutral placeholder content/images — nothing from the recording). Reuse the shared text
  constant that the concurrent change introduces; if it is inlined in `generate/llm_prompt.py`,
  P1-G extracts it to `phrasing.py` as a refactor with **byte-identical** transition snapshots.
- Layout: intro → `STRUCTURE` (viewport ≈W×H px; track of placeholder cards, pitch ≈p px if
  observed; content duplicated once for a seamless loop) → `BEHAVIOUR` numbered, only for
  observed behaviours, e.g.:
  1. "Autoplay: the track moves continuously to the right at ≈40 px/s with constant (linear)
     velocity and loops seamlessly (one loop ≈… ms)."
  2. "When the pointer hovers the scroller (or: is pressed on it; or: trigger not visible —
     hover or press), autoplay slows to a stop over ≈500 ms (ease-out)."
  3. "Dragging: the content follows the pointer 1:1 along the horizontal axis. Use Pointer
     Events with `setPointerCapture`; set `touch-action: pan-y` on the scroller so vertical page
     scrolling still works on touch; prevent text/image selection while dragging."
  4. "On release, continue with the pointer's release velocity, decaying exponentially with
     τ ≈ 200 ms (v·e^(−dt/τ), frame-rate independent). Measured release speeds ranged
     ≈500–1500 px/s; compute the release velocity from recent pointer movement, don't hardcode."
  5. Snap (grid): "then settle on the nearest card (step ≈216 px) over ≈300 ms (ease-out)" /
     abrupt-ambiguous: "the motion sometimes stops abruptly; this may be a snap or the pointer
     stopping before release — do not add snapping unless intended."
  6. "≈X ms after the motion comes to rest, autoplay resumes in its original direction, ramping
     from 0 to ≈40 px/s over ≈750 ms (ease-in-out)."
- `CONSTRAINTS`: estimates disclaimer (pixel ratio), wrap position modulo one copy's width,
  **`prefers-reduced-motion: reduce` → no autoplay (drag still works)**, pause autoplay when the
  tab is hidden, "do not invent behaviours not described", low-confidence items phrased with the
  existing hedges.

### 6.3 `css.py` (continuous) — "suggested implementation"
- Header comment as today. Layout: `.scroller { overflow: hidden; touch-action: pan-y; }`
  (+ `cursor: grab` / `:active { cursor: grabbing }` when drag observed), `.scroller__track
  { display: flex; width: max-content; }`.
- Autoplay via `@keyframes`: period observed → `to { translate: -{period}px 0 }`, duration
  `{loop.duration_ms}ms linear infinite`. Not observed → `translate: calc(-1px * var(--copy-width)) 0`
  and `animation-duration: calc(var(--copy-width) / {speed} * 1s)` with a comment "set
  `--copy-width` to one copy's width in px (unitless)". Direction right/down →
  `animation-direction: reverse` (no extra numbers). Vertical axis → `translate: 0 …`.
- `pause.on == hover` → `.scroller:hover .scroller__track { animation-play-state: paused; }` +
  comment "measured: slows over ≈N ms rather than stopping instantly — see JS tab".
- `@media (prefers-reduced-motion: reduce) { .scroller__track { animation: none; } }`.
- Drag/inertia/snap/resume present → comment "Autoplay + drag + momentum need the JS driver
  (JS tab); keep only the layout rules above." and the `@keyframes` block is omitted.

### 6.4 `js.py` (new `js` output; only in continuous mode)
- Header `// Suggested driver — estimated from a screen recording, not the original code.`
- Constants block from IR only: `AUTOPLAY_PX_S` (signed), `PAUSE_DECEL_MS`, `INERTIA_TAU_MS`
  (or tween duration), `SNAP_STEP_PX` / `SNAP_MS` (only when snap kind = grid), `RESUME_DELAY_MS`,
  `RESUME_RAMP_MS`; omitted behaviours → constants omitted and their code paths dropped.
- Fixed template (~60 lines): one `requestAnimationFrame` loop owns position; state machine
  `autoplay | decel | drag | inertia | snap | waiting | resume`; Pointer Events + capture;
  pointer-velocity estimate over the last `INERTIA_TAU_MS / 2`; `v *= Math.exp(-dt / INERTIA_TAU_MS)`
  (towards the autoplay velocity when the IR's `v_inf ≠ 0`, see P2d notes);
  wrap modulo one copy (see P2d notes); velocity ramps use a cubic-bezier helper with the
  IR's bezier; `matchMedia('(prefers-reduced-motion: reduce)')` disables autoplay;
  `visibilitychange` pauses.
- Literal numbers allowed outside IR values: `{0, 1, 2, 1000}` (structural / unit conversion),
  documented next to `CSS_IDENTITY` in the test.

### 6.5 Tests (`tests/unit/test_generators_continuous.py`)
Snapshot tests on the continuous fixture and three variants (autoplay-only marquee; drag+inertia
no cursor; drag+snap with cursor); determinism; number provenance for technical / llm_prompt /
css / js (reuse `allowed_numbers` / `unexplained_numbers` from `test_generators.py`); prompt
contains the HTML-file contract, `touch-action`, `prefers-reduced-motion`; no `@keyframes` when
drag present; uncertain (< 0.3) values only under "Uncertain observations".

---

## 7. UI (W-lib by coder, W-ui by ui-ux-master; data/states only)

### 7.1 `web/lib/velocity.ts` (new, pure, tested)
```
VelocityModel {
  window: {start_ms, end_ms}
  axis: "x"|"y"; unit: "px/s"; positiveLabel: "right"|"down"; negativeLabel: "left"|"up"
  points: {t_ms, v, pos, quality, degraded: boolean}[]    // from continuous.samples
  yDomain: {min, max}                                      // symmetric, padded
  suggestCompressedScale: boolean                          // max|v| / |v_auto| > 10 (asinh scale offered)
  autoplayRef: {v, band} | null                            // reference line
  bands: {phase_id, kind, label, start_ms, end_ms, band, uncertain, interrupted,
          headline: string /* e.g. "τ ≈ 200 ms", "≈ 1500 px/s peak" */}[]
  markers: {t_ms, kind: "cursor_enter"|"cursor_leave"|"release"|"rest"|"resume_start"}[]
}
```
Plus `format.ts` labels for every new enum member (`PHASE_KIND_LABELS`, interaction/pattern/
trigger/role/warning/error labels) and `errors.ts` entry for `continuous_motion_unsupported`;
mock fixture sync (`pnpm -C web sync:fixture` for the continuous fixture).

### 7.2 Components / states
| Component | Data | States |
|---|---|---|
| `VelocityChart` (new) | `VelocityModel`; playhead synced with `usePlayback`; click/Enter on band → seek to `start_ms`; hover/focus → phase details (kind, duration, v start→end, fit values + confidence) | autoplay-only (single band), degraded samples (visually distinct **not by color alone**, e.g. dashed/gap), low/uncertain bands (hatched + label), compressed-scale toggle, no samples (fallback to phase list), vertical axis wording |
| `ContinuousSummary` (new; replaces `InteractionSummary` in continuous mode) | `continuous.autoplay` + `behavior.*` as rows with `ConfidenceBadge`; trigger source (cursor vs velocity) | behaviour not observed (row hidden or "not observed"), pause trigger unknown, loop not observed, snap ambiguous |
| `ResultView` | `spec.mode` switch: continuous → `ContinuousSummary` + `VelocityChart`, no `MotionTimeline` | — |
| `KeyframeStrip` | `kind: "phase"` frames labelled by the phase at `t_ms` | missing images |
| `SpecTabs` | adds **JS** tab when `outputs.js != null` | — |
| `JobErrorState` / `describeError` | `continuous_motion_unsupported` title + guidance | — |
| `WarningsPanel` | new warning codes | — |

Accessibility: bands keyboard-navigable (same ARIA grid conventions as `MotionTimeline`),
`aria-live` for playhead phase change optional.

---

## 8. Synthetic ground truth (Track S)

### 8.1 Renderer extension (no change to existing renders)
- `tests/synth/kinematics.py` (new, **independent** of `pipeline/continuous/*`, own integrator):
  `VelocityProfile` built from phases `Autoplay(v, dur)`, `Ramp(v_from, v_to, dur, easing)`,
  `Hold(dur)`, `Drag(cursor-driven)`, `Inertia(tau, v_inf)` (v0 = cursor velocity at release,
  analytic), `SnapTween(step, dur, easing)`, `Stop`. Position by closed form where available, else
  numeric integration on a 0.1 ms grid. Wrap: `translate = −(x mod L)` for a content strip
  duplicated once.
- `scene.py`: `Scene.drivers: dict[node_id, Callable[[t], (tx, ty)]]` overriding timeline values;
  `state_key` includes driver outputs. Scenes without drivers render byte-identically
  (`test_generated_files_match_definitions` must stay green).
- `truth.py`: optional `continuous: TruthContinuous{axis, region, autoplay_velocity_px_s,
  loop_period_px|None, loop_observable, pitch_px, phases[{kind, start_ms, end_ms, v_start, v_end,
  tau_ms?, v0?, ramp_ms?, easing?, snap_step?}], pause_on, resume_delay_after_rest_ms,
  resume_ramp_ms, snap_kind}`.
- `evaluate.py`: `compare_continuous`, phase matching (same kind, max time-IoU), summary block;
  `oracle_spec` builds a valid continuous IR from truth (keeps `test_oracle_is_valid_ir_and_passes`
  meaningful).

### 8.2 Scenarios (canvas 1280×800; strip = 8 distinct procedural cards 200×160, gap 16 → pitch 216,
duplicated; viewport 760×200)
| # | Name | Content | Primary checks |
|---|---|---|---|
| C1 | `marquee_autoplay` | 60 px/s left, 8 s, idle cursor outside | speed, direction, mode continuous, loop **not observed** (travel 480 < period 1728) |
| C2 | `marquee_fast_loop` | 4-card loop (period 864), 240 px/s left, 8 s | loop period observed, loop duration |
| C3 | `marquee_hover_pause` | 40 px/s right; cursor enters 2.0 s → decel 400 ms ease-out to 0; leaves 4.0 s → delay 300 ms, ramp 600 ms ease-in | pause on hover, decel/ramp, resume delay |
| C4 | `carousel_drag_inertia` | 40 px/s; press 2.0 s (autoplay stops); drag left 400 px/450 ms; release v0≈−1100, τ 200 ms; rest; resume after 1000 ms, ramp 750 ms ease-in-out; second drag right, τ 200 ms | phase sequence, τ, v0, resume, pointer ratio |
| C5 | `carousel_drag_inertia_no_cursor` | C4 without cursor (the real-case analogue) | same values; pause trigger unknown |
| C6 | `carousel_drag_inertia_30fps` | C4 @ 30 fps | relaxed targets |
| C7 | `carousel_drag_inertia_vfr` | C4, `EncodeSpec(vfr=True)` | VFR timestamps |
| C8 | `carousel_drag_snap` | 2 drags; release → snap tween to nearest card 300 ms ease-out | snap grid, step, snap duration |
| C9 | `carousel_fling_fast_crf28` | peak drag ≈2500 px/s, τ 250 ms, crf 28, 30 fps | large-shift tracking, `tracking_degraded` allowed, values within relaxed targets |
| C10 | `ticker_vertical_autoplay` | vertical strip, 30 px/s up, hover pause | axis y |
| C11 | `hover_with_ambient_marquee` | S1 card hover + an autoplay-only marquee elsewhere from t=0 | **transition mode**, S1 §11.4 thresholds, warning `ambient_motion_masked` |
| C13 | `carousel_dark_edge_zoom` (P2b) | dark page = track #020202, low-contrast #141418 cards 164×150 (1 px border, inset photos, every 3rd dark), gap 4, cards scaled ×1→×1.165 by distance from the viewport centre; autoplay 35 → fling left whose momentum blends into autoplay → fling right → rest 200 ms → resume 815 ms; no cursor, 75 Hz grid MOV | card size / pitch / gap reported (required), inertia→autoplay, phase sequence |
| C14 | `carousel_flat_held_stop` (P2e) | dark page, flat placeholder cards (plain fill + title) 164×150, gap 4, rigid; autoplay 35 → fling left and fling right, each blending into autoplay → a drag that stops while pressed → held 200 ms → resume 815 ms material-decelerate; no cursor, 60 fps | row split into activity pieces (merged), drag ending at rest, rest → resume, no glide |
| N1 | `negative_ambient_pulse` | opacity-pulsing box from t=0 (chained tweens), no other motion | `continuous_motion_unsupported` |
| — | `negative_scroll` (existing) | unchanged | still `unsupported_motion` |

### 8.3 Unit fixtures (no video)
- `phases` on a velocity profile reconstructed **from the numbers in §0.1** (piecewise, noise
  added): expected sequence autoplay → decelerate(interrupted) → drag → inertia → drag → inertia →
  drag → inertia → drag → stop|snap → resume → autoplay. Numbers only; no frames or content.
- `displacement` on numpy-rendered shifted textures (sub-pixel shifts, shifts up to 120 px,
  box-blurred references, periodic textures for wrap ambiguity).
- `regime` on hand-built activity grids (lead-window cases, cursor removal, region split).

### 8.4 Accuracy targets (60 fps crf 18; 30 fps / VFR / crf 28 in parentheses)
| Metric | Target | Justification |
|---|---|---|
| mode + axis + autoplay direction | 100 % | discrete, must be exact |
| autoplay speed | ≤ 2 % (≤ 3 %) | ≥ 1 s of linear motion; position noise ≈0.1 px → < 0.5 %; codec/VFR margin |
| loop period (observable) | ≤ 1 % ; no false period when not observable | NCC on ≥ 0.5 crop overlap is sub-pixel; false periods are worse than "not observed" |
| pitch / snap step | ≤ 2 px | edge-profile periodicity, averaged over several cards |
| τ (τ ≥ 120 ms) | ≤ 15 % (≤ 25 %) | ≥ 6 samples per τ at 60 fps, fit window ~3τ → SE ≈ 5 %; ×3 margin; 30 fps halves samples |
| release velocity v0 | ≤ 15 % (≤ 25 %) | release time ±½ frame → bias ≈ e^(dt/2τ) ≈ 4 % at 60 fps / 8 % at 30 fps, plus kink smoothing |
| drag peak speed | ≤ 10 % (≤ 15 %) | instantaneous estimate, smoothed over 5 samples |
| sharp boundaries (drag start, release, snap/stop end) | ≤ 2 frames: 34 ms (67 ms) | kinks are detectable to ±1 frame; one frame margin |
| soft boundaries (decelerate / resume t0) | ≤ max(60 ms, 15 % of ramp) (max(90 ms, 20 %)) | ease-in starts are ill-defined; same joint t0/D fit as §6.7 |
| ramp / decel / snap duration | ≤ max(60 ms, 20 %) (max(90 ms, 25 %)) | consistent with §11.4 duration tolerance widened for velocity-domain ramps |
| resume delay | ≤ 50 ms (≤ 80 ms) | difference of a sharp and a soft boundary |
| phase sequence (merged same-kind) | exact on all positive scenarios | grammar output |
| snap detection | 100 % on C8, 0 false snaps on C1–C7, C9–C10 | false snaps produce wrong implementations |
| pause trigger | hover correct with cursor (C3, C10); `unknown` without (C5) | — |
| C11 transition values | all §11.4 thresholds | masking must not degrade the transition path |
| regression | existing 17 scenarios: IR identical except `schema_version`, `mode`, `meta.generated_at`/`pipeline_version`; `make eval` 17/17 | §3.2 rule 4 |
| performance | 15 s 1080p60 continuous recording < 30 s (dev Mac) | §11.4 budget |

---

## 9. Phases, tracks, file ownership, DoD

```
P0 contract + baseline (sequential, 1 coder)
 ├─► Track R  regime + masked scan          ┐
 ├─► Track C  continuous measurement        │ parallel, no file overlap
 ├─► Track S  synth kinematics + scenarios  │
 ├─► Track G  continuous generators         │
 ├─► Track W  W-lib (coder) ‖ W-ui (ui-ux-master, against the fixture)
 └─► Track I  interpreter wording           ┘
        ▼
P2 integration + calibration (sequential, 1 coder) ─► P3 UI validation (ui-ux-master)
        ▼
P4 hardening / optional (cursor-in-band, static-start drag scroller, perf)
```

### P0 — Contract + baseline (M, 1 coder, blocks all)
Files: `api/app/models/ir.py`, `api/app/models/measure.py`, `api/app/api/schemas.py`,
`api/app/core/errors.py`, `api/app/pipeline/params.py` (`ContinuousParams`),
`api/app/pipeline/persist.py` (allow-list), `api/app/generate/__init__.py` (Outputs.js +
dispatch + `generate/continuous/__init__.py` with stub modules raising `NotImplementedError`),
`api/app/contract.py`, `docs/contract/motion-spec.schema.json` (generated),
`docs/contract/sample-continuous-result.json` (new), `web/lib/types.ts` (generated),
`api/tests/unit/test_ir_contract.py` (+ continuous validator tests).
- Baseline: before any edit, run `make eval` and copy `data/synth/*.ir.json` to
  `data/synth/baseline/` (gitignored); add `scripts/compare_ir_baseline.py` (ignores
  `schema_version`, `mode`, `meta.generated_at`, `meta.pipeline_version`).
- **DoD:** `make contract-check`, `make lint`, `uv run pytest` green; transition fixture and
  continuous fixture validate; `pnpm -C web typecheck` fails only where new enum members need
  labels (expected, fixed in W-lib) — or W-lib labels land in the same PR to keep `master` green
  (preferred).

### P1 tracks (parallel after P0)
**Track R — regime + masked scan (M).** Files: `api/app/pipeline/scan.py`,
`api/app/pipeline/continuous/regime.py` (new), `api/tests/unit/test_regime.py` (new),
`api/tests/unit/test_scan*.py` (extend).
- DoD: `ambient=None` path byte-identical (unit test on recorded `Pass1Stats` of S1 + compare
  script on the suite); ambient detection returns no regions for all 17 existing scenarios
  (test over their pass-1 stats); masked segmentation unit-tested on synthetic energy with an
  ambient strip; `decide_mode` table fully covered.

**Track C — continuous measurement (L).** Files: `api/app/pipeline/decode.py` (`iter_region`
only), `api/app/pipeline/continuous/{displacement,panorama,kinematics,phases,confidence,measure,assemble}.py`
(new), `api/tests/unit/test_continuous_{displacement,panorama,kinematics,phases,assemble}.py` (new).
- DoD: displacement recovers sub-pixel and ≤ 120 px shifts within 0.1 px / 0.5 px on rendered
  textures incl. blurred references; kinematics fits recover τ/ramps from noisy synthetic
  profiles within §8.4 at 60/30 fps; the §0.1-derived profile yields the expected phase sequence;
  `assemble_continuous` emits a valid IR for the fixture-shaped measurement.

**Track S — synth (L).** Files: `api/tests/synth/{kinematics.py (new), scene.py, scenarios.py,
truth.py, evaluate.py}`, `api/tests/synth/test_synth_*.py`, `api/scripts/eval_synth.py` (summary
block), `Makefile` (only if a new target is needed).
- DoD: all C/N scenarios render + truth round-trips; existing scenario files byte-identical;
  `oracle_spec` for every C scenario validates and passes `compare`; a perturbed continuous IR
  fails.

**Track G — generators (M).** Files: `api/app/generate/continuous/{phrasing,technical,llm_prompt,css,js}.py`,
`api/app/generate/export_json.py`, `api/tests/unit/test_generators_continuous.py`,
`api/tests/snapshots/continuous_*` (new). Depends on the concurrent single-HTML-file change being
merged (only to reuse/extract its contract text; may touch `generate/phrasing.py` for that
extraction, nothing else in transition generators).
- DoD: §6.5 tests green; transition snapshots unchanged.

**Track W — web.** W-lib (coder): `web/lib/velocity.ts` (+ test), `web/lib/format.ts`,
`web/lib/errors.ts`, `web/lib/mock/*` (continuous fixture). W-ui (ui-ux-master):
`web/components/result/{VelocityChart,ContinuousSummary}.tsx` (new), `ResultView.tsx`,
`KeyframeStrip.tsx`, `SpecTabs.tsx`, `WarningsPanel.tsx`, error-state component.
- DoD: typecheck/lint/test/build green; mock mode renders the continuous fixture with all §7.2
  states (degraded, uncertain, autoplay-only, vertical) reachable via fixture variants.

**Track I — interpreter (S).** Files: `api/app/interpret/prompt.py` (continuous wording: label
the scroller and structure only), `api/app/interpret/payload.py` if keyframe naming needs it,
`api/tests/unit/test_interpret.py` (extend).
- DoD: fake-claude tests cover continuous inputs; schema still has no numeric fields except
  confidence.

### P2 — Integration + calibration (L, sequential, 1 coder)
Files: `api/app/pipeline/run.py` (`measure` branches: pass 1 → `detect_regime` → masked
`build_scan_result` / `analyse_ambient` → `decide_mode`; `apply_ambient_mask` on windows;
continuous `Measurement`), `api/app/pipeline/assemble.py` (dispatch to `assemble_continuous`),
`api/app/pipeline/keyframes.py` (phase keyframes `phase_k` at phase midpoints, ≤ 6, plus
annotated region), `api/app/pipeline/debug.py` (dump activity grid, displacement, phases),
`api/scripts/calibrate*.py` (continuous Monte Carlo over τ ∈ 80–600 ms, ramps 300–1000 ms,
speeds 20–2500 px/s, 60/30 fps, σ 0.05/0.2 px), `docs/PLAN.md` (one-line pointer to this file),
`README.md` (supported motion + recording tip).
- DoD: `make eval` = 17/17 existing (baseline compare clean) + all C/N scenarios within §8.4;
  `pytest -m synth` asserts the same; continuous bands ordered (high ≥ 90 % within targets);
  persist round-trip byte-identical for a continuous measurement; re-run interpretation works on
  a continuous job; E2E upload of C4 through the UI. **User smoke test**: the user runs
  `scripts/analyze.py` on the motivating recording and checks the phase sequence/values against
  §0.1 (no numeric gate; frames not viewed by the agent).

### P3 — UI validation (S, ui-ux-master)
Final visual review of continuous result page vs design system; screenshots for README from
synthetic C4 only.

### P4 — Hardening / optional (M)
- In-band cursor detection: motion-compensated residual `|I_t − shift(I_{t−1}, d_t)|` finds a
  pointer that does not move with content (hover/press, release divergence); during drag it moves
  with content and is inferred from entry/exit positions. Adds `evidence: velocity+cursor`.
- Static-start drag scroller (no autoplay, motion begins after a static lead): detect "viewport
  fixed, content translating > 0.5× viewport along one axis" inside a regime-A run and route to
  continuous analysis. Must keep the 17 existing scenarios on the transition path.
- Performance pass if P2 exceeds budget (½-res coherence test, NCC-only when q high).

---

## 10. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Wrap-around / repeating identical cards alias the shift when per-frame shift ≥ pitch/2 (≈3200 px/s at 30 fps, pitch 216) | constant-velocity prediction + acceleration clamp choose the peak; consistency check `d(i,i−2)`; `tracking_degraded` |
| Blur / compression smear at > 1500 px/s (real case response 0.55–0.7) | blur-matched retry, multi-scale, quality-weighted fits, C9 stress scenario |
| Cursor not recorded (real case) | velocity-only path is primary; trigger fields `unknown` with low confidence; abrupt stop reported as ambiguous |
| Cursor merges with the band in pass 1 | region-level analysis does not depend on cursor; P4 residual detection |
| Vertical scrollers | axis-agnostic code; C10 scenario |
| Multiple simultaneous scrollers / opposing rows | strip clustering split; largest interacted scroller wins; `extra_scrollers_ignored` |
| Page scroll mistaken for a scroller | area / frame-span rule → `unsupported_motion`; `negative_scroll` regression unchanged |
| Ambient detection false positives breaking existing behaviour | only motion present in the first 0.6 s counts; no-ambient path byte-identical; test over all 17 scenarios' pass-1 stats |
| VFR timestamps / recorder duplicate frames | passthrough timestamps, dups skipped (not zero velocity), position-domain fits |
| Inertia model mismatch (libraries use friction physics, CSS ease-out tweens, or springs) | fit exponential vs tween by BIC, report the model; overshoot flagged, spring not reconstructed |
| Snap vs "pointer stopped" ambiguity | snap requires grid evidence; else `abrupt_ambiguous` with explicit wording |
| Element animations inside the moving scroller (card hover lift while scrolling) | out of scope; not measured, mentioned in NOTES when residual energy inside the region is high |
| Generator number provenance for derived values (loop duration) | derived values are stored in the IR (`loop.duration_ms`), never computed in generators |
| Concurrent edit of `generate/llm_prompt.py` | continuous prompt lives in `generate/continuous/llm_prompt.py`; dispatch in `render_all`; only an extract-constant refactor touches shared code, after the other change merges |

## 11. Out of scope (this plan)
Two-axis/free panning; rotation/3D/coverflow carousels; nested scrollers; more than one analysed
scroller per recording; drag-and-drop of items; spring parameter reconstruction (overshoot
flag only); exact replay of the drag path; rubber-band/edge bounce of non-looping carousels;
stepped autoplay slideshows (advance one slide every N s — regime A analyses the first step as a
transition); element transitions inside a moving scroller; native page/momentum scroll
(`unsupported_motion`); scroll-driven animations; the static-start drag scroller (P4 optional).

## 12. Open questions — resolved by the user (2026-10-03)

1. **Mode override on upload:** **auto only** for now (no `JobOptions` field). Revisit if auto misroutes.
2. **PRD update:** **yes** — amend PRD §25/§26 in P0: single-axis scroller with autoplay, pause, drag,
   inertia, snap, resume is in scope; spring reconstruction stays out of scope.
3. **`js` output tab:** **yes**, continuous mode only (contract change to `Outputs` in P0).
4. **Re-recording with cursor:** still open (asked separately). Plan proceeds with the no-cursor path as
   primary; P4 cursor-in-band work stays optional.

Note: the single-HTML-file prompt contract landed on master in 239542a (`llm_prompt.OPENING_LINE`,
OUTPUT section). Track G reuses/extracts that text.

## 13. Addendum — measured appearance in the prompt (user decision 2026-10-03)

User request: the HTML the coding LLM builds should look as close as possible ("1:1") to the
video. Decisions: **content stays placeholder** (no text/images from the recording leave the
machine or land in outputs); the **prompt carries measured appearance values**; the HTML is still
written by the coding LLM from the prompt (Mimic does not generate HTML). Honest limit: fonts,
real images and exact text are not recoverable; expect "visually close", not pixel-identical.

**Contract (P0, `models/ir.py`, applies to both modes):**
- `ElementStatic` gains `background_color: MeasuredColor | None`, `text_color: MeasuredColor | None`
  (text-like elements only), `font_size_px: MeasuredNumber | None` (low-confidence estimate from
  text line height; capped at medium), alongside the existing `border_radius_px` and `shadow`.
- New optional top-level `scene`: `viewport_css` (w, h of the recorded frame in CSS px),
  `page_background: MeasuredColor | None`. Existing `bbox_initial` / `bbox_active` (CSS px) remain
  the geometry source; positions are reported relative to the parent (or the viewport for roots).
- Continuous mode: the scroller band box, card size, gap and pitch, card background — reuse the
  same `ElementStatic` fields on the scroller/card elements plus the existing pitch value.
- All new values carry confidence and are optional (absent = not measured). Transition IR without
  them stays valid; generators must render identically when they are absent (snapshot-safe).

**Measurement (new Track A, P1; files: `api/app/pipeline/appearance.py` (new),
`api/tests/unit/test_appearance.py` (new); hook call in `pipeline/assemble.py` — Track A is the
only P1 track touching `assemble.py`, P2 continues from there):**
- Sample state A frames (full-res pass-2 crops or the state_a keyframe) inside each element bbox:
  background = dominant colour of an inner ring just inside the border (excluding child bboxes and
  text pixels); text colour = dominant minority cluster in text-like boxes (k-means k=2 in Lab);
  page background = mode of pixels outside all element bboxes; font size ≈ text line box height ×
  ratio (report low confidence). Colour confidence from cluster purity and H.264 chroma error
  (cap like the existing colour cap 0.85). Validate on the synthetic suite where truth is known
  (Track S adds appearance truth for existing scenes without changing their videos).

**Generators (Track A, transition generators only: `generate/llm_prompt.py`, `technical.py`,
`css.py`; do not touch `generate/phrasing.py` — Track G may extract shared text there):**
- LLM prompt: new **APPEARANCE (measured, approximate)** block after STRUCTURE: viewport size,
  page background, per element size and position relative to its parent, background colour,
  text colour, radius, shadow, approximate font size — each with its hedge per confidence band.
  OUTPUT keeps placeholders for content but says to match these measured sizes/colours exactly,
  and to keep the measured desktop layout at the recorded viewport width while still reflowing
  responsively below it.
- CSS: base rules gain width/height/background/color/border-radius/box-shadow from the IR.
- Technical: an APPEARANCE section.
- Number-provenance invariant must keep passing (all new numbers come from the IR).

## P0 notes (2026-10-03, contract + baseline landed)

Gaps/contradictions found while implementing P0 and how they were resolved. Tracks read this
before §5.

**Serialization / compatibility**
- Fields new in 0.2 that a transition spec does not use are **omitted from the JSON when null**
  (`Field(exclude_if=...)`): `MotionSpec.continuous`, `MotionSpec.scene`,
  `ElementStatic.background_color|text_color|font_size_px`, `Outputs.js`. `types.ts` marks them
  optional (`continuous?: ContinuousMotion | null`). `mode` is always serialized (4th key, after
  `meta`). Result: pipeline IR for all 17 scenarios differs from the baseline only in
  `schema_version`, `mode`, `meta.generated_at` (also inside `outputs.json`); `outputs.js` never
  appears in transition mode.
- Transition text outputs (technical / llm_prompt / css) are byte-identical. The **JSON-tab
  snapshots** (`tests/snapshots/{sample,low_confidence}.json.json`) changed by exactly two lines
  (`"schema_version": "0.2"`, `"mode": "transition"`): unavoidable, the export is the IR.
- `docs/contract/sample-result.json` bumped to 0.2 + `mode`; its `outputs.json` regenerated; new
  test keeps fixture outputs == `render_all(spec)`. Web mock synced (`pnpm sync:fixture`).
- **Stored 0.1 results are served verbatim** by `GET /api/jobs/{id}/result` (no `mode` key) while
  `types.ts` says `mode` is required. W-lib must read the mode via a helper that defaults to
  `"transition"` (e.g. `specMode(spec)`); a re-run (`/interpret`) re-dumps and adds `mode`.
- `persist.FORMAT_VERSION` stays **1**: `Measurement.continuous` defaults to `None`, verified with
  a v1 document without the key (`test_persist.py`).
- `meta.pipeline_version` not bumped (P2 decides when measurement behaviour changes).

**Contract additions / tightenings beyond §5.2 (all in `models/ir.py`)**
- Named models: `LoopInfo`, `Span`, `Size`, `MeasuredColor`, `Scene`; fit models `ConstantFit`,
  `ExponentialFit`, `RampFit`, `TweenFit` (discriminator `model`); behaviour models
  `PauseBehavior`, `DragBehavior`, `InertiaBehavior`, `SnapBehavior`, `ResumeBehavior`.
- `ContinuousMotion.gap_px: MeasuredNumber | None` added (§13 needs gap; generators must not
  derive it from pitch − card width).
- Extra validators: phase ids exactly `p1..pN` in order; phases inside `span_ms`; behaviour ⇔
  phases (`drag` set ⇔ a drag phase and `count` = #drag phases; `inertia` ⇔ inertia phase and
  `instances` = #inertia phases; snap `grid` ⇒ a `snap` phase, `abrupt_ambiguous` ⇒ a `stop`
  phase and no `step_px`; `resume` ⇒ a resume phase); `interaction.type == "drag"` ⇔
  `behavior.drag`; continuous mode needs `schema_version "0.2"` and `target_element_id` = the
  scroller; `loop.duration_ms` = period / speed · 1000 (±1 ms); samples strictly increasing in
  `t_ms`, quality 0..1; `font_size_px` confidence < 0.8; `ResultEnvelope`: `outputs.js` set ⇔
  continuous mode.
- `PHASE_FIT_MODELS` (kind → allowed fit) and `SIGN_CONVENTION` are module constants;
  `PHASE_KINDS`, `CONTINUOUS_SAMPLE_MAX`, `SIGN_CONVENTION`, `SCHEMA_VERSION = "0.2"` are
  exported to `types.ts`.
- Cards inside the scroller (appearance, §13) are ordinary elements: `role: "card"`,
  `kind: "transform"`, `parent_id` = scroller (no new `ElementKind`); text inside a card likewise
  (`text_like: true`). See the continuous fixture (e1 scroller, e2 card, e3 title).
- Keyframes: `kind: "phase"`, names `phase_1.png` .. `phase_99.png`.

**Interpreter guard (touches Track I's file minimally)**: `interpret/schema.py` `ROLES` /
`INTERACTION_TYPES` exclude the continuous-only members (`scroller`, `continuous`, `drag`), so
Layer B cannot put them into a transition spec (the IR would reject it). Track I decides what the
continuous schema offers.

**Generators**: `render_all` dispatches on `spec.mode` (`GENERATOR_MODULES` /
`CONTINUOUS_GENERATOR_MODULES`, the latter = `continuous.{technical,llm_prompt,css,js}` +
shared `export_json`). The four continuous modules are stubs raising `NotImplementedError`, so
`render_all` on a continuous spec raises until Track G. `export_json` already drops
`continuous.samples` (done in P0, Track G need not touch it). The continuous fixture's text/js
outputs are placeholders; its `json` output is the real export.

**Internal models (`models/measure.py`)**: `RegimeReport` does not carry the cursor (it stays in
`ScanResult.cursor`; avoids persisting `CursorTrack`) and has `cell_css`, `grid_shape`,
`lead_window_s`. `ScrollerAnalysis` has `reason: NotScrollerReason | None`
(`two_axis | untrackable | opposing_unsplit`, selects the §3.5 message variant), `gap_px`, and
`behavior` as the IR `Behavior` model; `PhaseCandidate.fit` is the IR `PhaseFit` (same pattern as
`FittedTransition.easing`). `INTERACTION_PHASE_KINDS` backs `ScrollerAnalysis.has_interaction`.
`Measurement.continuous` lives in `pipeline/assemble.py` (where `Measurement` is defined).

**Fixture**: `docs/contract/sample-continuous-result.json` uses the §8.2 synthetic geometry
(1280×800, viewport 760×200 at 260,300, cards 200×160, pitch 216, gap 16), not the real
recording's; its velocity profile follows §0.1: autoplay +39 → decelerate (interrupted by the
drag) → 4 drags with 3 inertia phases (τ 200/210/190 ms, two re-grabbed) → abrupt `stop`
(snap `abrupt_ambiguous`) → `paused` 100 ms (the grammar needs `paused` between stop and resume)
→ resume ramp 750 ms → autoplay. No cursor, VFR, loop not observed. 187 samples (50 ms).

**Baseline**: `data/synth/baseline/*.ir.json` (gitignored via `data/`) captured before any edit;
`api/scripts/compare_ir_baseline.py` also parses `outputs.json` and ignores `outputs.js` when
null/absent. P0 result: 17/17 clean.


## 14. Acceptance goal — round-trip replica (user definition, 2026-10-03)

User: the work is done when the motivating recording (local only, not in the repo) becomes an
HTML page whose **CSS, JS behaviour and flow match the video**; assets (images/text) may differ.

**How it is proven (objective, no frame viewing by agents):**
1. `scripts/analyze.py` on the recording → IR + LLM prompt.
2. A coding LLM builds `index.html` **from the prompt only** (no video, no keyframes). For the
   automated run this is a sub-agent in the orchestrating session; the user can also paste the
   prompt into any coding LLM and feed the result to step 3.
3. **Round-trip harness** `api/scripts/roundtrip.py` (new; uses the existing headless Chrome
   available to `~/.claude/tools/screenshot/cdp_flow.mjs` — no new dependency): open the HTML at
   the recorded viewport (CSS px, same pixel ratio), replay the interaction derived from the IR
   (wait through autoplay; for each measured drag phase dispatch Pointer/mouse events along the
   axis with the measured displacement and duration so the release velocity matches; hover
   enter/leave when a hover pause was measured), capture frames with real timestamps (CDP
   screencast or deterministic begin-frame capture) → encode with ffmpeg → analyze.
4. **Compare** source IR vs replica IR with `scripts/compare_roundtrip.py` (new) and print a
   pass/fail table.

**Pass criteria (continuous mode):** identical phase-kind sequence (allowing merge of phases
shorter than 2 frames); autoplay speed ±5 % and same direction; inertia τ ±20 %; resume delay and
ramp duration ±100 ms; pause/decel duration ±100 ms when present (ramps compared by the time to
reach 90 % of the speed change, P2e); snap presence identical; scroller box and card
size/pitch/gap ±2 px (card scaling ±0.03 at the source's reference, P2e); background/text colours
ΔE ≤ 5 where measured.
Transition mode (bonus, same harness): per-transition duration ±20 ms or 10 %, value within
§11.4 targets of PLAN.md, easing family identical.

**Phasing:** after P2 (integration + calibration) and P3 (UI). Also run the harness on the
synthetic C4/C8/C3 and S2 scenarios first, so the harness itself is validated on known truth
before it is pointed at the user's recording. The user may additionally compare visually
(side-by-side output written locally for the user only).


## P2 notes (2026-10-04, integration + calibration)

**Wiring (`pipeline/run.py`).** `measure`: pass 1 (`scan.pass1_stats`) → `detect_regime` →
`scan_for_regime` (masked when ambient). No ambient region → the transition path exactly as before.
Ambient → `analyse_ambient` (pointer from `scan.masked_cursor_track`) → `decide_mode`; transition:
`apply_ambient_mask` on every pass-2 decode (also the regrow); continuous: `measure_continuous`
(heuristic = `continuous_heuristic`, cursor events vs the scroller, card/title layout + appearance
from one rest frame, keyframe plan). `assemble.assemble` / `run_interpreter` dispatch on
`Measurement.continuous`; `continuous/assemble.py` passes `scene`. Keyframes (continuous):
`state_a`, `annotated_a` (scroller/card/title boxes), `phase_<k>` at ≤ 6 phase midpoints.
Debug dumps: `activity.csv`, `masks/activity_mean.png` (cell grid, not a frame), `regime.json`,
`scroller_<i>.json`, `displacement_<i>.csv`, `layout.json`. `PIPELINE_VERSION` 0.2.0.
`scripts/analyze.py`: `--print js`, `--no-keyframes` (writes no frame image at all), phase table.

**Appearance (§13) wired for both modes.** Transition: `measure_appearance` on the pass-2 state-A
crop + the full state-A frame (keyframe size; reused for `state_a.png`, byte-identical).
Continuous: `continuous/layout.py` finds, in one still frame, the viewport (moving band grown over
the visible track background), the first whole card (gap columns = track colour on every band row,
folded by the pitch) and its title line; the frame-based gap (pitch − card) replaces the panorama's
(which blur widened: C9 31 → 16 px). `appearance.background_color` retries thinner rings (inset
card images). Eval judges reported colours (fill ΔE ≤ 5, text ≤ 10) and the card box (±2 px).

**Pointer over a scroller.** The pass-1 tracker cannot see a pointer over moving content
(`cursor.MAX_BIG_FRAC`). The continuous path tracks it with the ambient regions masked
(`scan.masked_cursor_track`) and `continuous/pointer.occlusion_corrected` places the unseen samples
inside the region when the pointer vanished at the mask edge heading in. Gives hover vs press
(C3/C10 hover; C4 press: autoplay → drag while the pointer had been inside for longer than a
hover reaction) and `ResumeBehavior.delay_after_leave_ms`. **Not** measured: pointer ↔ content
ratio during drags (the pointer moves with the content): reported unknown, and the evaluator no
longer judges `drag.pointer_ratio` when the IR says unknown (a wrong reported ratio still fails).
In-band detection stays P4.

**Unified inertia end (Track S ↔ Track C).** A decay to rest ends where the fitted speed reaches
**10 px/s** (the synthetic truth's cut), unless the content is observed at rest earlier while the
fit is still ≥ 15 px/s **and** a cut beats the uncut decay by Δχ² ≥ 25 (`phases._inertia_end`;
noise used to fake early cuts at σ 0.2 px). The end speed is reported as
`ExponentialFit.stop_px_s` / `InertiaBehavior.stop_px_s` and the JS driver stops there, so the
resume delay runs on the same clock in source and replica.

**IR additions (optional, omitted when null; `make contract`).** `ExponentialFit.stop_px_s`,
`InertiaBehavior.stop_px_s`, `ResumeBehavior.delay_after_leave_ms`; `Scene.page_background`
omitted when null. Not done: an explicit `offset` field (the derived parent offset stays
allow-listed by the provenance test; adding it would touch every transition element).

**Other measurement changes.** Pause behaviour also for autoplay → drag (press stops at once,
`decel_ms` null). Tween releases start at a continuous fitted time (snap release was +44 ms) and
the drag ends at its own velocity. Ramp onsets may lie inside a run of skipped duplicate frames
(C9 second resume: 527 ms linear → 751 ms ease-in-out). Drag onset searched further ahead. Short
phases: drags last ≥ 2 frames after the onset, rests < 2 frames are re-grabs, slowdowns cut by a
press within 2 frames are the press. Drag peaks: Theil–Sen velocity over ≥ 100 ms, low-quality
samples left out, never a stop's one-step speed; peak confidence ≤ tracking quality around it.
Inertia whose τ is off the scroller's (weighted median of ≥ 3 instances) by > 2× is the pointer
slowing down and merges into the drag. A coherent region that only rests is not a scroller
(`testsrc` → `continuous_motion_unsupported`). `iter_region` / the tracker report a run of
duplicate frames (a clean encode at rest) every 100 ms and at the end. `pitch()` guards
non-positive autocorrelation peaks. Pitch folding uses `FOLD_EPS` (111/216·216 = 110.999…).
`scan.py`: OpenCV 5 `phaseCorrelate` windows its inputs in place → copies (no IR change).

**Capture-grid regression.** The motivating `.mov` has stamps on a 13.3 ms grid while the page
renders at 60 Hz, so content positions lag their stamps by 0..1 render interval. Synth
`EncodeSpec(capture_grid_hz=75)` reproduces it (C12 `carousel_drag_inertia_grid75`, MOV, no
cursor) and unit fixtures `make_grid_series`. Both within §8.4 (VFR targets).

**Suite / calibration.** `CONTINUOUS_SUITE_ENABLED = True`: `make eval` 30/30 (17 transition +
13 continuous), baseline compare 17/17 clean (`compare_ir_baseline.py` strips the §13 appearance
additions and re-renders the outputs from the stripped spec, proving everything else is
byte-identical; `card_hover_webm` did not drift). Eval prints a continuous calibration block;
`pytest -m synth` asserts high ≥ 90 % for both. `scripts/calibrate_continuous.py` (in
`make calibrate`): 160 Monte Carlo profiles (τ 80–600 ms, peaks 300–2500 px/s, ramps 300–1000 ms,
60/30 fps, σ 0.05/0.2, ¼ on the capture grid): high band 97 % within target (resume delay 91 %,
the weakest: eased onsets in σ 0.2 noise).

**Motivating recording (numbers only, no frames viewed or stored).** autoplay +35.3 px/s →
drag 2.15–2.56 s → inertia τ 262 ms (v0 −1409, decays towards autoplay) → drag 3.59–5.40 →
rest → drag right 5.54–6.16 → glide 303 ms → rest → slow drag 6.58–6.68 → inertia τ 258 →
drag left 7.04–7.44 → glide 244 ms to rest 7.69 → rest 195 ms → resume ramp 815 ms → autoplay
+34.9 px/s. Pitch / loop not observed (no cards in the layout), pause trigger unknown (no
pointer), drag peak ≈2920 px/s at medium confidence (`tracking_degraded`).

**Open for P3 / later.** UI: show `stop_px_s` / `delay_after_leave_ms`; ContinuousSummary words an
instant press-stop as "slows to a stop". §14 round-trip harness not built yet. P4 in-band pointer.


## P2b notes (2026-10-04, card detection on the motivating recording)

**Diagnosis (numbers only; no frame of the recording was viewed or stored).** The earlier
"no cards" was not low contrast: the cards are photo-filled (L* ≈ 36–42 against the page's 1.3;
some photos are dark, L* ≈ 12) with ≈4–5 px gaps of page colour. Three things broke the
panorama pitch:
1. **The cards scale with their distance from the scroller centre.** Per-frame segmentation of
   288 frames: ≈164×149 px at the centre, ≈192×176 px at ≈270 px from it, quadratic in the
   distance (×1.17); one card tracked through the leading autoplay shrinks 179×165 → 170×156 as
   it moves 75 px towards the centre while another grows 173 → 185. Visible gaps stay ≈4–5 px,
   so centre-to-centre spacing varies 172 (centre) – 178 px. Content is not a rigid translation.
2. Consequently the stitched panorama smears (per-column spread of the ≤ 4 contributions:
   median 19, p75 83 gray levels) and the edge-energy autocorrelation is dominated by photo
   texture: no peak near the card spacing (top lags 1632 / 1621 / 1605 px).
3. Even single frames, without stitching, are not periodic: `panorama.pitch` on 288 single
   frames found a "pitch" on 16 (all wrong, 185–296 px). `layout.py` needs a pitch → no card.

**Fix: card census (`continuous/cards.py`).** Every pasted frame of the tracking box's rows
(over the whole crop width — the coherence-window box can be narrower than the scroller where
low-contrast content barely changed) is segmented on its own: track background = the band's
cross margins (else the commonest flat-column level); gap columns = within clip(4σ, 3, 8) gray
levels of it on ≥ 95 % of the rows; a card = the run between two gap runs, both edges seen,
≥ 24 px, not within 4 px of the scroller ends (where page = track a clipped card looks short);
sub-pixel edges from the per-row coverage of the edge column; cross extent likewise. Aggregate:
size vs distance from the **scroller centre** fitted as `L0 + b·d²` on distance-bin medians;
zoomed when it grows > 3 % over the observed range (cross size agreeing) → card = the centre card
(`L0 × C0`), pitch = `L0 + gap`, gap = median gap near the centre, confidence capped at 0.6
(one size stands for a range); rigid → medians, pitch = median centre spacing, grid phases from
the card edges. `measure.card_layout` keeps the panorama unless it has no pitch, the cards scale,
or its pitch is a whole multiple of a confident census pitch. `ScrollerAnalysis` gained
`card_len_px / card_cross_px / card_zoom_per_px2` (defaulted floats, persist format unchanged);
`layout.find_card_by_census` segments the rest frame, takes the card nearest the centre that
fits the expected (zoomed) size and reports it at the census size centred where it is drawn;
`run.py` no longer overrides the census gap with `pitch − card`. With zoomed cards the pitch is
not used as snap-grid evidence (tracker units are an average over differently magnified
content). Evaluator: a continuous scroller without pitch / gap / card now **fails**.

**Kinematics found by C13 (same recording family).** (a) `absorb_short` relabelled runs in
sequence order, so a just-relabelled one-sample blip made the next short autoplay run look
isolated → autoplay lost / drag start 480 ms early; now shortest run first, merging after every
change. (b) A 35 px/s autoplay on the 75 Hz grid wobbles by σ ≈ 2.7 px/s (beat of 75 Hz stamps vs
60 Hz frames), more than the 4.2 px/s autoplay band: the band is now ≥ 3 σ of the leading
autoplay's velocity noise (≤ 30 % of the speed). (c) Momentum that blends into autoplay after a
fling **against** the autoplay direction crosses zero, so it was labelled inertia → paused →
resume: `_merge_blend` refits one exponential (v∞ = v_auto) from the release through the resume
and merges when reduced χ² ≤ 4, τ within 30 % and the standstill ≤ 150 ms (note "momentum blends
back into autoplay"). Synth truth: an inertia with v∞ ≠ 0 is not a rest.

**C13 truth for zoomed cards.** `Strip(zoom_edge=…)` drives every card with
`kinematics.EdgeZoom` (x = tan(√a·u)/√a, s = 1 + a x²) through a new optional driver output
(`(tx, ty, scale)`); card / gap / pitch truth = the geometry at the centre (gap 3.12 px: the lens'
convexity narrows it), speeds × the mean viewport magnification (1.055, `TruthZoom`). Measured:
card 164.3×150.1 (truth 164×150), pitch 167.1 (167.12), gap 2.8 (3.12), zoom per px² 1.14e-6
(1.143e-6), speed 2.0 % off (3 % target: the tracker weights the centre a little more than the
uniform mean). `make synth`: all existing videos and truths byte-identical except the VP9 webm.

**Motivating recording after P2b (numbers only).** Card (centre) ≈162.8×147.2 px, gap ≈5.1 px,
pitch ≈167.8 px (medium 0.60 / gap 0.52), cards ×1.17 at 261 px from the centre (849 cards in
288 frames); scroller 744×186 at (91, 305), page / scroller #020202; card fill not measured
(photo-filled cards, no plain ring). Momentum decays towards autoplay (τ ≈ 260 ms). Two
tween-shaped glides to rest (244 / 303 ms, easeOutQuad) are consistent with snap-to-card but
are not claimed: the rest positions cannot be checked against the card grid while the cards
scale.

**Resolved in P2c** (below): `ContinuousMotion.card_scale` carries the scaling into the IR and
every output.


## P2c notes (2026-10-04, position-dependent card scaling)

**What the recording does (numbers only, census of 849 cards in 288 frames).** Card size grows
quadratically with the on-screen distance of its centre from the scroller centre (×1.17 at
261 px) while the **visible gaps stay ≈5 px everywhere** (4.8–5.3 px from the centre out to
200 px) and the centre-to-centre spacing grows with the cards (171 → 184 px). So the cards are
scaled **and stay packed** — not scaled in place with a rigid layout (that would make
neighbours overlap by ≈15 px near the edges). C13's lens (`EdgeZoom`) is the same family (it
scales the gaps as well; the census fits the same `k`).

**Contract (`models/ir.py`, optional, omitted when null).** `ContinuousMotion.card_scale:
CardScale {model: "quadratic", origin: "scroller_center", reference_distance_px,
scale_at_reference, mean_scale, confidence}`: `scale(d) = 1 + (scale_at_reference − 1)·(d /
reference_distance_px)²`, ×1 at the centre by definition (the card box, `pitch_px`, `gap_px`
are the centre card's). Stored by its value at the farthest whole card measured (readable
numbers in every output; `k ≈ 2.5e-6` would not survive the provenance check).
`mean_scale = 1 + (s_ref − 1)(half / ref)² / 3` (half = half the region length) is the mean
magnification over the scroller: the IR's speeds / displacements are on-screen values, an
unscaled track moves at `speed / mean_scale` (recording: 1.114 — without it a replica runs
≈11 % fast, outside §14's ±5 %). Validators: reference inside the region (±1 px), scale > 1,
`mean_scale` consistent (±1e-3). No `center_scale` / `axis` fields: both are implied.

**Measurement → IR.** `CensusResult.zoom_confidence` = card count × size spread × how clearly
the growth clears the 3 % threshold (full at 6 %); `card_layout` / `ScrollerAnalysis` carry
`card_zoom_reach_px` + `card_zoom_confidence`; `assemble.card_scale` maps them (reference
clamped to half the region) with `ContinuousParams.cap_card_scale = 0.6` (medium, same
reasoning as the zoomed pitch). Recording: `{reference 261.3 px, ×1.1694, mean 1.1144,
medium 0.6}`. Evaluator (C13): scale at the IR's reference distance vs the truth lens ±0.03,
`mean_scale` vs `TruthZoom.speed_scale` ±0.02 (the centre is ×1 by definition, its size is
the card box check ±2 px); a rigid truth reporting a scale fails.

**Generators.** Technical: "Card scaling" row (+ pitch / card size "at the scroller centre").
Prompt: BEHAVIOUR item after Autoplay (formula from IR numbers, per-frame update, packed layout,
speeds ÷ `mean_scale`); STRUCTURE / APPEARANCE say the pitch and card size are the centre's;
OUTPUT: update scales in the same rAF loop with translate + scale (no reflow). CSS: header,
`position: relative` on scroller / track, `will-change` + a comment on the card. JS:
`SCALE_REF_PX`, `SCALE_AT_REF`, `SPEED_SCALE`; `scaleCards(offset)` packs the cards outward
from the centre (closed-form quadratic per card, layout gaps kept, `offsetLeft` ignores
transforms), `x += v·dt / SPEED_SCALE`, drag `Δp / SPEED_SCALE`, copy size from layout
offsets (scaled cards must not count; see P2d notes). Uncertain scale (< 0.3): only under the
uncertain sections, not implemented. Rigid cards: every output byte-identical.
UI: `ContinuousSummary` "Card scaling" row (only when measured), mock variant
`continuous_glide` carries one.


## P2d notes (2026-10-04, wrap period + momentum wording)

**Wrap period.** The JS driver wrapped modulo half the track length. With a flex `gap` the track
holds `2N` cards and `2N − 1` gaps, so half of it is half a gap short of one copy and the track
jumped by `gap / 2` on every loop (2.5 px on the recording, gap ≈5 px). One copy is now the
distance from the first card of copy A to the first card of copy B (`children[n / 2]` −
`children[0]`, the gap between the copies included): rigid cards via `getBoundingClientRect`
(sub-pixel; the track's translate cancels out), scaled cards via `offsetLeft` / `offsetTop`
(layout, so the cards' own translate / scale do not count; whole px). Node test
`test_js_wrap_is_seamless` runs autoplay / fast drags over ≥3 loops (rigid + scaled) and
asserts the offset only ever jumps by exactly one copy and no wrap frame moves a card further
than an ordinary frame; a control re-inserts the old formula and sees the 8 px (gap 16 / 2) jump.

**Momentum towards autoplay.** When every exponential inertia fit decays towards a non-zero
`v_inf` (the recording: +34.7 px/s ≈ autoplay; `cp.inertia_blends_into_autoplay`), no output
says "comes to rest": prompt `v = v_auto + (v − v_auto) · e^(−dt/τ)`, "blends back into autoplay,
which simply keeps running, so no rest, delay or resume ramp follows such a release" (+ "Resume
below applies only after the motion has come to rest" when a resume was measured; the hover pause
re-arms on re-enter; reduced motion: decays to zero); technical `Decays towards:` row (no `Stops
below`); JS `vEnd = canAutoplay() ? autoV : 0`, hand-over to `state = 'autoplay'` once < 1 px is
left (`rest()` only when decaying to zero). `_no_resume` then only talks about a pause. Decay to
zero: unchanged ("until it comes to rest").

**Drag with card scaling.** The unscaled track moves `Δpointer / mean_scale`, so the content keeps
up with the pointer on average, not card by card (the unmagnified centre card moves ≈1/mean_scale
of the pointer, edge cards slightly more). The prompt says "follows the pointer 1:1 on screen on
average" (never "exactly") with that caveat; JS unchanged (`(p - lastP) / SPEED_SCALE`).


## P2e notes (2026-10-04, §14 acceptance iteration 1)

Run 1 (replica built by an independent agent from the prompt only) failed 9/16. Root causes, all
found numerically (the recording's frames were not viewed):

**1. Releases — one law, not two.** Model comparison on the recording's displacement series
(grid search + linear LSQ, BIC): after release 1 the decay is towards the autoplay velocity
(exp → v_auto τ 262 ms, rms 0.39 px; exp → 0 rms 1.48 px, ΔBIC 157); after the 110 ms standstill
the +146 px/s flick also decays towards autoplay (τ 256, rms 0.04 vs 0.11 px). The two "glides to
rest" (p7 303 ms, p12 244 ms after "releases" at 1570 / 2350 px/s) are not momentum: their
deceleration time constant is ≈60 ms (τ 260 fixed: rms 3–5× worse), the momentum law would need
≈1.3–1.4 s to slow to 10 px/s, and p12's speed was still rising 80 ms after the supposed
release (−1945 → −2340 px/s). They are the pointer slowing to a stop **while pressed** (the same
thing P2 already merged for p4 via τ inconsistency). The resume after the last one is a genuine
eased ramp (material-decelerate 725–815 ms beats an exponential approach from rest, ΔBIC 28; τ 260
fixed is far worse), after a ≈195 ms standstill. So the page has: fling → momentum blending into
autoplay; let go without a fling (pointer still) → rest → ≈195 ms → resume ramp. "Towards
autoplay" is real, not an artefact. The user's "glides then continues autoplay" is the drag's own
slowdown followed by the resume.
- `phases._merge_pointer_stops`: a tween glide after a drag that ends in < ½ of the time the
  scroller's exponential momentum (≥ 2 instances agreeing within 2×) needs from the same speed to
  10 px/s merges into the drag (`NOTE_POINTER_STOP`); the drag ends at rest.
- `_unify_momentum_target`: an interrupted (re-grabbed) decay fitted towards zero is refitted
  towards v_auto when another release visibly decays towards autoplay (ΔBIC ≤ 6).
- Resume delays: a resume after a drag that ended at rest has no `delay_after_release_ms` (the
  lift is not visible).
- Outputs: `cp.drag_stops` / `cp.resumes_after_drag_stop`. Prompt: release = fling (momentum
  blends into autoplay) vs zero-velocity release (no momentum → Resume counted from the last
  movement, never while pressed; "not a separate glide"). JS: `movedAt`, `vRelease === 0` →
  `rest()` with `restAt = movedAt`. Technical: "Released at rest" line.
- `cp.pause_by_press_inferred`: trigger not visible + no slowdown phase + drags → the only pause
  evidenced is the press; prompt / JS / technical / CSS describe a press pause and no hover pause
  (run 1's builder added a hover pause because the prompt told it to).

**2. Geometry — Mimic mis-measured the replica** (layout itself was also wrong: centred instead of
x 91, y 305, the prompt never asked for the position in OUTPUT). Flat placeholder cards only
change at edges / titles during the lead window, so §3.2 split the row into 3 ambient regions;
the largest piece (176 px) was "the scroller", with pitch 190 and no card.
- `regime.merge_row_pieces` (+ re-analysis in `run.measure`, masked scan and pointer recomputed):
  analysed regions with the same axis, ≥ 60 % cross overlap, same autoplay direction and speeds
  within 1.5× merge into one region spanning the frame along the axis (`AmbientRegion.merged`);
  `displacement.changed_span` widens the tracking box to the first … last changed column (a
  piece-sized box lost fast drags and read speeds off a few, differently magnified features).
- The reported scroller box is unpadded (`refine_box(pad=0)`): the 2 px working pad made every
  box 4 px too wide / tall, i.e. the source's IR said 744×186 for a 740×182 band and a faithful
  replica measured 4 px off. Scroller colour: when no track shows around the cards (box = band),
  sampled just outside the band (`outer_ring_color`).
- Prompt: APPEARANCE "top-left corner at x 93, y 307 … (not centred)", OUTPUT layout keeps the
  measured position at the recorded viewport size; CSS uses `margin: 307px 0 0 93px` (+ `body {
  margin: 0 }`) for a non-centred scroller. CSS: `will-change: translate` on the track, none on
  the cards (fixed-scale card layers re-rasterised mid-motion jittered ±1 px).

**3. Spurious 353 ms "decelerate" at the replica start** — the piece-sized tracking box (2.): the
left-edge piece's local speed varies with the card scaling. Also the harness entered the
scroller one frame before pressing; now enter + press share a frame.

**4. Card size / pitch like-for-like** — `card.scale` row (replica scale at the source's
reference ±0.03); scaling on one side only fails and card size / pitch are then `n/a` (centre
card vs uniform size).

**Harness / compare.** Drags whose release speed is < 30 px/s end at rest: a rest that follows is
held still (through it into the next drag as one press, else `RELEASE_HOLD_MS` 150 ms, longer
than the τ/2 velocity window, then let go); their pointer path replays the content's own measured
path (IR `samples`, ±30 ms smoothing, ±60 ms for degraded drags, unimodal speed, 3-frame taper
to rest); flings keep the validated parametric model. Eased ramps (resume, pause slowdown) are
compared by the time to reach 90 % of the speed change (an easeOutQuart over 1080 ms and a
material-decelerate over 815 ms reach it within 25 ms of each other; their "durations" differ by
260 ms).

**Segmentation robustness on low-texture replicas** (tracking wobble ±0.5–1.5 px): blips up to
2 px (or 5 local σ) off the autoplay line stay autoplay, also at the end of the recording;
decelerate / resume ramps tolerate single noisy samples (≥ 85 % in a noise-widened band) and end
where the speed reaches the band for good; a decelerate prefix must leave the autoplay line; drag
onsets are measured against the autoplay line of the last 0.5 s and need ≥ 0.3 px; a slowdown
cut by a press within 3 frames is the press.

**Regression.** C14 `carousel_flat_held_stop` (flat placeholder cards, rigid; fling left / right
blending into autoplay; a drag that stops while pressed → 200 ms → resume 815 ms; synth
`HeldDrag`, `Strip(flat=True)`). Not in C14: flat cards that also scale make the tracker
under-read the mean on-screen speed by ≈4 % (> 2 % target; documented limitation). `make eval`
32/32, baseline 17/17.

**Recording after P2e:** autoplay +35.3 → drag → momentum (τ 262 → v_auto) → drag (stops with the
pointer) → paused 133 → drag (stops) → paused 110 → drag (flick 117 px/s) → momentum (τ 258 →
v_auto, re-grabbed) → drag (stops) → paused 195 → resume 815 ms material-decelerate → autoplay;
scroller 740×182 at (93, 307), card 162.8×147.2, pitch 167.8, gap 5.1, ×1.17 at 261 px. A replica
built from Mimic's own CSS + JS tabs (identical flat, distinct flat and photo-like placeholders)
round-trips 18/18.

## Acceptance result

**§14 acceptance: PASS (2026-10-04, run 3, 18/18 judged rows).** The replica was written by an
independent coding agent that read only Mimic's LLM prompt for the motivating recording (no video,
keyframes or IR) and was never hand-edited; every failing run was fixed on Mimic's side and the
page rebuilt from the new prompt. The recording's frames were not viewed by agents.

| Run | Verdict | Root cause | Fixed in Mimic |
|---|---|---|---|
| 1 | FAIL 9/16 | Release modelling: every release was treated as momentum blending into autoplay, while the page also lets go without a fling (pointer still → rest → resume). Geometry: flat placeholder cards split the replica's row into ambient pieces, so a 176 px piece was "the scroller"; the 2 px working pad made boxes 4 px too large; the prompt never gave the scroller's position. Spurious decelerate at the replica start (piece-sized tracking box) | P2e: `_merge_pointer_stops`, `_unify_momentum_target`, zero-velocity release in prompt / JS / technical, `pause_by_press_inferred`; `merge_row_pieces` + `changed_span`; unpadded scroller box; APPEARANCE position + non-centred CSS; enter + press on one frame in the harness |
| 2 | FAIL 11/17 | The prompt's card-scaling rule let the builder extrapolate the quadratic curve to off-screen cards: far from the centre the packed layout has no solution and one card grew far larger than the scroller and covered it | Prompt: apply the curve only to cards at least partly inside the scroller, leave the others unscaled, never extrapolate off screen; JS driver and technical text do the same (`test_generators_continuous.py` pins the wording) |
| 3 | **PASS 18/18** | — | — |

Run 3, source vs replica: phase sequence identical; autoplay 35.3 vs 33.9 px/s; momentum τ 258 vs
260 ms, decaying towards autoplay; resume delay 195 vs 197 ms, ramp 815 vs 841 ms; scroller
740×182 at (93, 307) vs 741×182 at (92, 307); card 162.8×147.2 vs 162.7×146.8; pitch 167.8 vs
168; gap 5.1 vs 5.3; card scale ×1.169 vs ×1.170 at 261 px; colours ΔE 0.3.

Artifacts (prompt, source IR, replica `index.html`, replica video, `compare.json`, harness log)
are in `data/acceptance/`. They are derived from the user's private recording, stay local only
(`data/` is gitignored) and must not be copied into tracked files.

Harness self-check (`make roundtrip` without arguments, synthetic truth, re-run 2026-10-04):
`harness self-check: PASS` — C4 and C3 pass, the deliberately wrong C4 replica fails as expected,
C8 and S2 fail one known measurement-limit row each (scroller box ±3 px from H.264 spill; thin-text
colour timing). Residual
limits are listed in the README's "Known limitations" (inferred triggers without a visible
pointer, flat scaling cards ≈3–4 % slow, box repeatability ±2–3 px, rough drag peak speeds).
