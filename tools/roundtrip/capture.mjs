#!/usr/bin/env node
// Round-trip capture driver (PLAN-continuous §14): open an HTML page in headless Chrome at a
// fixed viewport / device pixel ratio, replay timed mouse input, and capture one PNG per frame.
//
//   node capture.mjs <plan.json>
//
// No dependencies: Node's built-in fetch/WebSocket talk to the DevTools protocol of a browser this
// script launches itself (`--remote-debugging-port=0`, so it never collides with a debugging port
// already in use, and it only ever terminates the process it started).
//
// plan.json (written by api/scripts/roundtrip.py):
// {
//   "chrome": "/path/to/chrome-headless-shell", "headless_flag": "--headless" | "--headless=new" | null,
//   "url": "file:///.../index.html", "viewport": {"w": 1280, "h": 800}, "dpr": 1,
//   "fps": 60, "frames": 630, "warmup_ms": 500, "capture": "virtual" | "screencast",
//   "draw_cursor": false, "out_dir": "/tmp/.../frames",
//   "events": [{"t_ms": -500, "type": "move", "x": 640, "y": 700, "buttons": 0}, ...]
// }
// Event types: move (mouseMoved), down (mousePressed, left), up (mouseReleased, left).
// Events with t_ms < 0 are sent during the warm-up (e.g. parking the pointer).
//
// capture "virtual" (default, deterministic): Chrome virtual time is paused and advanced by exactly
// one frame interval per frame (JS timers, Date.now and performance.now follow it). An init script
// (INIT below) makes the rest of the page clock-consistent with it:
//   * requestAnimationFrame callbacks are queued and run once per captured frame with the frame's
//     virtual time as timestamp (one rAF per frame, like a 60 Hz display);
//   * CSS animations / transitions / Web Animations are driven from the same clock: each frame,
//     every animation is held at playbackRate 0 and its currentTime is set to its own elapsed
//     virtual time (honouring animation-play-state: paused and page-set playback rates);
//   * Event.timeStamp of input events reads the virtual clock at dispatch.
// Without this, CSS animations and event timestamps in headless Chrome follow wall-clock time,
// which drifts with the screenshot cost (measured: ~2x too fast at 36 ms/frame wall time).
//
// capture "screencast" (fallback, real time): Page.startScreencast frames with their compositor
// timestamps, input dispatched on a wall-clock schedule; frames.json then carries real (VFR)
// timestamps. Not deterministic; use when virtual time is unavailable.
//
// Writes <out_dir>/f_000000.png ... and <out_dir>/frames.json:
//   {"capture": "virtual", "fps": 60, "frames": [{"i": 0, "t_ms": 0.0, "file": "f_000000.png"}, ...],
//    "console": [...], "exceptions": [...], "events_sent": N, "wall_ms": ...}
// Exit code 0 on success; non-zero with a message on stderr otherwise.
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

const planPath = process.argv[2];
if (!planPath) {
  console.error('usage: capture.mjs <plan.json>');
  process.exit(2);
}
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const outDir = plan.out_dir;
fs.mkdirSync(outDir, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------
// Init script (runs before any page script, in every document)
// ---------------------------------------------------------------------------------------------
const INIT = `(() => {
  if (window.__rt) return;
  // Virtual time is exact, but performance.now() is coarsened with random jitter (~0.1-0.2 ms),
  // which made repeated captures differ. After start() the page clock is driven by the harness:
  // setTime(ms) pins it to the exact frame time; in between (timers firing while virtual time
  // advances) it reads frame time + whole ms elapsed since then. Every run sees the same times.
  const origNow = performance.now.bind(performance);
  let clockBase = null, lastExact = 0, lastOrig = 0;
  Object.defineProperty(performance, 'now', { configurable: true, writable: true, value: () => {
    if (clockBase === null) return origNow();
    const d = origNow() - lastOrig;
    return clockBase + lastExact + (d < 0.5 ? 0 : Math.round(d));
  } });
  const now = () => performance.now();
  // rAF: queue, run once per captured frame from __rt.frame()
  let rafQueue = new Map(); let rafId = 0;
  // Keep native frames flowing: with virtual time paused, a page that produces no damage stops
  // producing compositor frames, and Input.dispatchMouseEvent / Page.captureScreenshot then wait
  // forever (reproduced on a static page). A fully transparent 1x1 px element nudged by 0.01 px
  // on every native frame provides the damage; it never paints a pixel.
  const nativeRaf = window.requestAnimationFrame.bind(window);
  let pumpEl = null, pumpK = 0;
  const pump = () => {
    if (!pumpEl && document.documentElement) {
      pumpEl = document.createElement('div');
      pumpEl.setAttribute('aria-hidden', 'true');
      pumpEl.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;pointer-events:none;background:transparent;z-index:-2147483647';
      document.documentElement.appendChild(pumpEl);
    }
    if (pumpEl) { pumpK ^= 1; pumpEl.style.transform = 'translateX(' + (pumpK * 0.01) + 'px)'; }
    nativeRaf(pump);
  };
  nativeRaf(pump);
  window.requestAnimationFrame = (cb) => { const id = ++rafId; rafQueue.set(id, cb); return id; };
  window.cancelAnimationFrame = (id) => { rafQueue.delete(id); };
  // Event.timeStamp on the virtual clock (stamped at dispatch by a capturing listener)
  const desc = Object.getOwnPropertyDescriptor(Event.prototype, 'timeStamp');
  Object.defineProperty(Event.prototype, 'timeStamp', { configurable: true, get() {
    if (this.__rtTs === undefined) { try { this.__rtTs = now(); } catch (e) { return desc.get.call(this); } }
    return this.__rtTs; } });
  const stamp = (e) => { if (e.__rtTs === undefined) e.__rtTs = now(); };
  for (const t of ['pointerdown','pointermove','pointerup','pointercancel','pointerover','pointerout',
                   'pointerenter','pointerleave','gotpointercapture','lostpointercapture','mousedown',
                   'mousemove','mouseup','mouseover','mouseout','mouseenter','mouseleave','click',
                   'wheel','touchstart','touchmove','touchend','touchcancel','dragstart'])
    window.addEventListener(t, stamp, { capture: true, passive: true });
  // pointer events delivered to the page (the driver waits for them before rendering a frame:
  // mouse moves are rAF-aligned and acknowledged before the page has seen them)
  let seen = 0;
  for (const t of ['pointermove', 'pointerdown', 'pointerup', 'pointercancel'])
    window.addEventListener(t, () => { seen++; }, { capture: true, passive: true });
  // Optional pointer sprite (only when the plan asks for it)
  let cursorEl = null;
  const drawCursor = (x, y) => {
    if (!cursorEl) {
      cursorEl = document.createElement('div');
      cursorEl.setAttribute('aria-hidden', 'true');
      cursorEl.style.cssText = 'position:fixed;left:0;top:0;width:14px;height:20px;pointer-events:none;z-index:2147483647;' +
        'background:#000;clip-path:polygon(0 0,0 100%,28% 76%,50% 100%,64% 93%,43% 70%,100% 70%);' +
        'filter:drop-shadow(0 0 1px #fff) drop-shadow(0 0 1px #fff);';
      (document.body || document.documentElement).appendChild(cursorEl);
    }
    cursorEl.style.transform = 'translate(' + x + 'px,' + y + 'px)';
  };
  // Animations on the virtual clock. Every animation is held paused by the harness and its
  // currentTime is set from its own elapsed virtual time each frame (setting currentTime also
  // completes the pending pause synchronously, so no frame is ever waited for). Because pause()
  // makes Blink ignore animation-play-state, the CSS play state is read from computed style, and
  // the page's own pause()/play() calls are recorded through wrappers.
  const AP = Animation.prototype;
  const origPause = AP.pause, origPlay = AP.play;
  const tracked = new WeakMap();
  let started = false;
  AP.pause = function () { const s = tracked.get(this); if (s) s.userPaused = true; return origPause.call(this); };
  AP.play = function () { const s = tracked.get(this); if (s) s.userPaused = false; return origPlay.call(this); };
  const cssPaused = (a) => {
    if (typeof a.animationName !== 'string' || !a.effect || !a.effect.target) return false;
    try {
      const cs = getComputedStyle(a.effect.target, a.effect.pseudoElement || null);
      const names = cs.animationName.split(',').map((x) => x.trim());
      const states = cs.animationPlayState.split(',').map((x) => x.trim());
      const i = names.lastIndexOf(a.animationName);
      if (i < 0 || !states.length) return false;
      return states[i % states.length] === 'paused';
    } catch (e) { return false; }
  };
  // A running transform/opacity animation gets its own compositor layer in Chrome (smooth
  // sub-pixel motion). Held animations are painted on the main thread instead, where e.g. rounded
  // rects snap to 1/4 px. So while the harness holds such an animation, its target gets
  // will-change (restored afterwards), mirroring the layer a running animation would have.
  const COMPOSITED = /^(transform|translate|scale|rotate|opacity|filter|all)$/;
  const promoted = new Map(); // element -> original inline will-change
  const animatesCompositable = (a) => {
    if (typeof a.transitionProperty === 'string') return COMPOSITED.test(a.transitionProperty);
    try {
      return a.effect.getKeyframes().some((k) => Object.keys(k).some((p) => COMPOSITED.test(p)));
    } catch (e) { return false; }
  };
  const promote = (targets) => {
    for (const el of targets) {
      if (promoted.has(el)) continue;
      promoted.set(el, el.style.willChange);
      el.style.willChange = 'transform, opacity';
    }
    for (const [el, orig] of promoted) {
      if (targets.has(el)) continue;
      el.style.willChange = orig;
      promoted.delete(el);
    }
  };
  const syncAnimations = (vt) => {
    let n = 0;
    const targets = new Set();
    const list = document.getAnimations ? document.getAnimations() : [];
    for (const a of list) {
      let s = tracked.get(a);
      if (!s) {
        if (a.playState === 'finished') continue;
        // before start(): keep the progress it made while loading; after: it was created by
        // this frame's input / rAF callbacks, i.e. at the current virtual time
        s = { elapsed: !started && a.currentTime != null ? a.currentTime : 0, last: vt,
              userPaused: a.playState === 'paused' };
        tracked.set(a, s);
      }
      n++;
      if (s.comp === undefined) s.comp = animatesCompositable(a);
      if (s.comp && a.effect && a.effect.target instanceof Element) targets.add(a.effect.target);
      const rate = a.playbackRate;
      const running = !s.userPaused && !cssPaused(a);
      if (running) s.elapsed += (vt - s.last) * rate;
      s.last = vt;
      if (a.playState !== 'paused') { try { origPause.call(a); } catch (e) {} }
      const t = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      const end = t ? t.endTime : Infinity;
      if (running && rate > 0 && Number.isFinite(end) && s.elapsed >= end) {
        try { a.currentTime = end; a.finish(); } catch (e) {}
        continue;
      }
      try { a.currentTime = Math.max(0, s.elapsed); } catch (e) {}
    }
    promote(targets);
    return n;
  };
  window.__rt = {
    drawCursor,
    seen: () => seen,
    start() {
      clockBase = origNow(); lastOrig = clockBase; lastExact = 0;
      const n = syncAnimations(now()); started = true; return n;
    },
    setTime(ms) { lastExact = ms; lastOrig = origNow(); return clockBase + ms; },
    frame() {
      // browser frame order: update animations, then animation-frame callbacks; a second sync
      // adopts animations the callbacks (or this frame's input) created, at this frame's time
      const vt = now();
      syncAnimations(vt);
      const q = rafQueue; rafQueue = new Map();
      for (const cb of q.values()) { try { cb(vt); } catch (e) { setTimeout(() => { throw e; }); } }
      const n = syncAnimations(vt);
      return { t: vt, animations: n };
    },
  };
})();`;

// ---------------------------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------------------------
const profile = fs.mkdtempSync(path.join(plan.tmp_dir || '/tmp', 'mimic-rt-profile-'));
const args = [
  ...(plan.headless_flag ? [plan.headless_flag] : []),
  '--remote-debugging-port=0',
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--hide-scrollbars',
  '--mute-audio',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--disable-extensions',
  '--font-render-hinting=none',
  '--disable-lcd-text',
  'about:blank',
];
const chrome = spawn(plan.chrome, args, { stdio: ['ignore', 'ignore', 'pipe'] });
let chromeExited = false;
chrome.on('exit', () => { chromeExited = true; });
const cleanup = () => {
  if (!chromeExited) { try { chrome.kill('SIGTERM'); } catch {} }
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
};
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(130); });
process.on('SIGTERM', () => { cleanup(); process.exit(143); });

const wsUrl = await new Promise((resolve, reject) => {
  let buf = '';
  const timer = setTimeout(() => reject(new Error('browser did not report a DevTools endpoint within 20 s')), 20000);
  chrome.stderr.on('data', (d) => {
    buf += d.toString();
    const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) { clearTimeout(timer); resolve(m[1]); }
  });
  chrome.on('exit', (code) => reject(new Error(`browser exited early (code ${code}): ${buf.slice(-400)}`)));
}).catch((e) => { console.error(String(e.message || e)); process.exit(3); });

const ws = new WebSocket(wsUrl);
let msgId = 0;
const pending = new Map();
const listeners = new Set();
const consoleLines = [];
const exceptions = [];
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const p = pending.get(m.id);
    pending.delete(m.id);
    m.error ? p.reject(new Error(`${p.method}: ${JSON.stringify(m.error)}`)) : p.resolve(m.result);
    return;
  }
  if (m.method === 'Runtime.consoleAPICalled' && consoleLines.length < 200) {
    consoleLines.push(`${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 300)}`);
  }
  if (m.method === 'Runtime.exceptionThrown' && exceptions.length < 50) {
    const d = m.params.exceptionDetails;
    exceptions.push((d.exception?.description ?? d.text ?? '').slice(0, 400));
  }
  for (const l of listeners) l(m);
});
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve);
  ws.addEventListener('error', () => reject(new Error('DevTools websocket failed')));
});
const DEBUG = !!process.env.RT_DEBUG; // RT_DEBUG=1: log every DevTools call + animation states
const send = (method, params = {}, sessionId) =>
  new Promise((resolve, reject) => {
    const id = ++msgId;
    if (DEBUG) console.error(`> ${method} ${JSON.stringify(params).slice(0, 120)}`);
    pending.set(id, { resolve, reject, method });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
const waitFor = (method, sessionId, timeoutMs = 30000) =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(() => { listeners.delete(l); reject(new Error(`timed out waiting for ${method}`)); }, timeoutMs);
    const l = (m) => {
      if (m.method === method && (!sessionId || m.sessionId === sessionId)) {
        clearTimeout(timer);
        listeners.delete(l);
        resolve(m);
      }
    };
    listeners.add(l);
  });

const { browserContextId } = await send('Target.createBrowserContext');
const { targetId } = await send('Target.createTarget', { url: 'about:blank', browserContextId });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const S = (method, params) => send(method, params, sessionId);
const evaluate = async (expression) => {
  const r = await S('Runtime.evaluate', { expression, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`evaluate failed: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
  return r.result.value;
};

const W = plan.viewport.w;
const H = plan.viewport.h;
const fps = plan.fps || 60;
const nFrames = plan.frames;
const warmupMs = plan.warmup_ms ?? 500;
const mode = plan.capture || 'virtual';
const events = [...(plan.events || [])].sort((a, b) => a.t_ms - b.t_ms);

await S('Page.enable');
await S('Runtime.enable');
await S('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: plan.dpr || 1, mobile: false });
await S('Emulation.setEmulatedMedia', {
  features: [
    { name: 'prefers-reduced-motion', value: 'no-preference' },
    { name: 'prefers-color-scheme', value: plan.color_scheme || 'light' },
  ],
});
await S('Emulation.setFocusEmulationEnabled', { enabled: true });
if (mode === 'virtual') await S('Page.addScriptToEvaluateOnNewDocument', { source: INIT });
else if (plan.draw_cursor) {
  await S('Page.addScriptToEvaluateOnNewDocument', { source: INIT.replace('window.requestAnimationFrame =', 'window.__rtRafUnused =').replace('window.cancelAnimationFrame =', 'window.__rtCafUnused =') });
}

const loaded = waitFor('Page.loadEventFired', sessionId, 30000);
const nav = await S('Page.navigate', { url: plan.url });
if (nav.errorText) { console.error(`navigation failed: ${nav.errorText}`); process.exit(4); }
await loaded;
// fonts / images in real time before the clock is taken over
await evaluate('document.fonts ? document.fonts.ready.then(() => true) : true').catch(() => true);
await sleep(200);

const BUTTON = { move: 'none', down: 'left', up: 'left' };
const TYPE = { move: 'mouseMoved', down: 'mousePressed', up: 'mouseReleased' };
let evIdx = 0;
let sent = 0;
let expectSeen = 0; // pointer events the page should have received so far
let lastPos = null;
const dispatch = async (e) => {
  // a move to the current position with the same buttons produces no pointer event: skip it
  if (e.type === 'move' && lastPos && lastPos.x === e.x && lastPos.y === e.y && lastPos.buttons === (e.buttons || 0)) return;
  const params = {
    type: TYPE[e.type],
    x: e.x,
    y: e.y,
    button: e.type === 'move' ? (e.buttons ? 'left' : 'none') : BUTTON[e.type],
    buttons: e.type === 'down' ? 1 : e.type === 'up' ? 0 : (e.buttons || 0),
    clickCount: e.type === 'move' ? 0 : 1,
    pointerType: 'mouse',
  };
  await S('Input.dispatchMouseEvent', params);
  if (plan.draw_cursor) await evaluate(`window.__rt && window.__rt.drawCursor(${e.x}, ${e.y})`).catch(() => {});
  sent++;
  expectSeen++;
  lastPos = { x: e.x, y: e.y, buttons: e.type === 'down' ? 1 : e.type === 'up' ? 0 : (e.buttons || 0) };
};
const settleInput = async () => {
  // wait (real time) until the page has handled every dispatched pointer event
  if (mode !== 'virtual') return;
  const until = Date.now() + 1000;
  while (Date.now() < until) {
    const n = await evaluate('window.__rt ? window.__rt.seen() : -1');
    if (n < 0 || n >= expectSeen) return;
    await sleep(2);
  }
  console.error(`warning: page saw fewer pointer events than dispatched (${expectSeen})`);
  expectSeen = await evaluate('window.__rt.seen()');
};
const dispatchUntil = async (tMs) => {
  let any = false;
  while (evIdx < events.length && events[evIdx].t_ms <= tMs + 1e-6) { await dispatch(events[evIdx++]); any = true; }
  if (any) await settleInput();
};

const frames = [];
const t0wall = performance.now();
const shotParams = { format: 'png', optimizeForSpeed: true };

if (mode === 'virtual') {
  await S('Emulation.setVirtualTimePolicy', { policy: 'pause' });
  const advance = async (ms) => {
    if (ms <= 0) return;
    const expired = waitFor('Emulation.virtualTimeBudgetExpired', sessionId, 60000);
    await S('Emulation.setVirtualTimePolicy', { policy: 'advance', budget: ms });
    await expired;
  };
  const frameMs = 1000 / fps;
  await evaluate('window.__rt.start()');
  // warm-up: frames without screenshots (layout, fonts, autoplay start); t < 0 events go here
  const warmFrames = Math.round(warmupMs / frameMs);
  // exact cumulative schedule in microseconds (no drift); page clock = warm-up + at(k)
  const at = (k) => Math.round((k * 1000000) / fps) / 1000;
  const warmMs = at(warmFrames);
  for (let k = -warmFrames; k < 0; k++) {
    await evaluate(`window.__rt.setTime(${at(k + warmFrames)})`);
    await dispatchUntil(k * frameMs);
    await evaluate('window.__rt.frame()');
    await advance(at(k + warmFrames + 1) - at(k + warmFrames));
  }
  await evaluate(`window.__rt.setTime(${warmMs})`);
  await dispatchUntil(-1e-3);
  const base = (await evaluate('performance.now()'));
  for (let k = 0; k < nFrames; k++) {
    await evaluate(`window.__rt.setTime(${warmMs + at(k)})`);
    await dispatchUntil(at(k));
    const r = await evaluate('window.__rt.frame()');
    if (DEBUG) console.error('anims', JSON.stringify(await evaluate('document.getAnimations().map(a => [a.animationName || a.transitionProperty, a.playState, a.pending, a.currentTime, a.startTime])')));
    const shot = await S('Page.captureScreenshot', shotParams);
    const file = `f_${String(k).padStart(6, '0')}.png`;
    fs.writeFileSync(path.join(outDir, file), Buffer.from(shot.data, 'base64'));
    frames.push({ i: k, t_ms: Math.round((r.t - base) * 1000) / 1000, file, animations: r.animations });
    if (k < nFrames - 1) await advance(at(k + 1) - at(k));
  }
} else {
  // real-time screencast fallback
  const got = [];
  const onFrame = (m) => {
    if (m.method !== 'Page.screencastFrame' || m.sessionId !== sessionId) return;
    got.push({ data: m.params.data, ts: m.params.metadata.timestamp });
    S('Page.screencastFrameAck', { sessionId: m.params.sessionId }).catch(() => {});
  };
  listeners.add(onFrame);
  await S('Page.startScreencast', { format: 'png', everyNthFrame: 1, maxWidth: Math.round(W * (plan.dpr || 1)), maxHeight: Math.round(H * (plan.dpr || 1)) });
  // warm-up events (t < 0) right away, then the schedule on the wall clock
  await dispatchUntil(-1e-3);
  await sleep(warmupMs);
  const durMs = (nFrames * 1000) / fps;
  const start = performance.now();
  let startTs = null;
  while (performance.now() - start < durMs) {
    const t = performance.now() - start;
    if (startTs === null && got.length) startTs = got[got.length - 1].ts;
    await dispatchUntil(t);
    await sleep(2);
  }
  await dispatchUntil(Infinity);
  await sleep(100);
  await S('Page.stopScreencast');
  listeners.delete(onFrame);
  const firstTs = startTs ?? (got.length ? got[0].ts : 0);
  let k = 0;
  for (const g of got) {
    const tMs = (g.ts - firstTs) * 1000;
    if (tMs < 0 || tMs > durMs) continue;
    const file = `f_${String(k).padStart(6, '0')}.png`;
    fs.writeFileSync(path.join(outDir, file), Buffer.from(g.data, 'base64'));
    frames.push({ i: k, t_ms: Math.round(tMs * 1000) / 1000, file });
    k++;
  }
}

let finalEval = null;
if (plan.final_eval) finalEval = await evaluate(plan.final_eval).catch((e) => `error: ${e.message}`);
const meta = {
  capture: mode,
  final_eval: finalEval,
  fps,
  viewport: { w: W, h: H },
  dpr: plan.dpr || 1,
  frames,
  events_sent: sent,
  events_total: events.length,
  console: consoleLines,
  exceptions,
  wall_ms: Math.round(performance.now() - t0wall),
};
fs.writeFileSync(path.join(outDir, 'frames.json'), JSON.stringify(meta, null, 1));
console.log(`captured ${frames.length} frames (${mode}) in ${meta.wall_ms} ms; ${sent}/${events.length} input events; ${exceptions.length} page exceptions`);
// graceful shutdown: close the browser, wait for it to exit, then remove its profile
await Promise.race([send('Browser.close').catch(() => {}), sleep(2000)]);
ws.close();
for (let i = 0; i < 60 && !chromeExited; i++) await sleep(50);
if (!chromeExited) { try { chrome.kill('SIGKILL'); } catch {} await sleep(200); }
for (let i = 0; i < 5; i++) {
  try { fs.rmSync(profile, { recursive: true, force: true }); break; } catch { await sleep(200); }
}
process.exit(frames.length ? 0 : 5);
