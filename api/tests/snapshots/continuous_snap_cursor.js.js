// Suggested driver — estimated from a screen recording, not the original code.
// All values are approximate (display scale assumed 2x).
// Markup: .product-carousel > .product-carousel__track holding the cards twice (two identical copies back to back).
// Signed velocities: positive = content moves right (x) / down (y).

const AUTOPLAY_PX_S = -40; // linear autoplay to the left, high confidence
const PAUSE_DECEL_MS = 400; // slowdown to a stop on hover, high confidence
const PAUSE_EASE = [0, 0, 0.58, 1]; // ease-out
const SNAP_STEP_PX = 216; // card pitch, high confidence
const SNAP_MS = 300; // settle on the nearest card, medium confidence
const SNAP_EASE = [0, 0, 0.58, 1]; // ease-out
const RESUME_DELAY_MS = 400; // after the motion comes to rest, medium confidence
const RESUME_RAMP_MS = 600; // ramp back to autoplay speed, medium confidence
const RESUME_EASE = [0.65, 0, 0.35, 1]; // cubic-bezier(0.65, 0, 0.35, 1) (close to easeInOutCubic)
const RESUME_PX_S = -40; // autoplay velocity after resuming, same direction

const scroller = document.querySelector('.product-carousel');
const track = scroller.querySelector('.product-carousel__track');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

let x = 0; // track offset in px, unwrapped (wrapped only when rendering)
let v = 0; // velocity in px/s
let autoV = AUTOPLAY_PX_S; // current autoplay velocity
let state = 'autoplay'; // autoplay | decel | paused | drag | inertia | snap | waiting | resume
let last = performance.now();
let ramp = null; // velocity ramp or position tween: { from, to, ms, curve, start }
let restAt = 0;
let pressed = false;
let hovered = false;
let dragged = false;
let startP = 0;
let lastP = 0;

const lerp = (a, b, t) => a + (b - a) * t;
const bez = (p1, p2, t) =>
  lerp(lerp(lerp(0, p1, t), lerp(p1, p2, t), t), lerp(lerp(p1, p2, t), lerp(p2, 1, t), t), t);
// cubic-bezier easing: find the curve parameter for progress p by bisection
function ease([x1, y1, x2, y2], p) {
  let lo = 0;
  let hi = 1;
  while (hi - lo > 1 / 1000) {
    const mid = (lo + hi) / 2;
    if (bez(x1, x2, mid) < p) lo = mid;
    else hi = mid;
  }
  return bez(y1, y2, (lo + hi) / 2);
}
const progress = (now) => Math.min(Math.max((now - ramp.start) / Math.max(ramp.ms, 1), 0), 1);
const canAutoplay = () => !reducedMotion.matches && !scroller.matches(':focus-visible');
// one copy = first card of copy A -> first card of copy B, gap between the copies
// included (both move with the track, so its translate cancels out)
const copySize = () => {
  const kids = track.children;
  const b = kids[Math.floor(kids.length / 2)];
  return kids.length > 1
    ? b.getBoundingClientRect().left - kids[0].getBoundingClientRect().left
    : 0;
};
const pointer = (e) => e.clientX;

function start(kind, from, to, ms, curve) {
  state = kind;
  ramp = { from, to, ms, curve, start: performance.now() };
}

function rest() {
  v = 0;
  restAt = performance.now();
  state = pressed || hovered ? 'paused' : 'waiting';
}

function snapTo(target) {
  const to = Math.round(target / SNAP_STEP_PX) * SNAP_STEP_PX;
  v = 0;
  start('snap', x, to, SNAP_MS, SNAP_EASE);
}

function letGo() {
  if (state === 'decel' && canAutoplay()) start('resume', v, RESUME_PX_S, RESUME_RAMP_MS, RESUME_EASE);
  else if (state === 'decel' || state === 'paused') rest();
}

function frame(now) {
  const dt = (now - last) / 1000;
  last = now;
  if (state === 'autoplay') {
    v = canAutoplay() ? autoV : 0;
  } else if (state === 'decel') {
    const p = progress(now);
    v = lerp(ramp.from, ramp.to, ease(ramp.curve, p));
    if (p === 1) {
      v = 0;
      state = 'paused';
    }
  } else if (state === 'resume') {
    const p = progress(now);
    v = canAutoplay() ? lerp(ramp.from, ramp.to, ease(ramp.curve, p)) : 0;
    if (p === 1) {
      autoV = ramp.to;
      state = 'autoplay';
    }
  } else if (state === 'snap') {
    const p = progress(now);
    x = lerp(ramp.from, ramp.to, ease(ramp.curve, p));
    if (p === 1) {
      x = ramp.to; // land exactly on the grid
      rest();
    }
  } else if (state === 'waiting') {
    if (now - restAt >= RESUME_DELAY_MS && canAutoplay()) {
      start('resume', 0, RESUME_PX_S, RESUME_RAMP_MS, RESUME_EASE);
    }
  }
  x += v * dt; // v is zero while dragging, snapping, gliding or paused
  const size = copySize();
  if (size > 0) track.style.translate = `${((x % size) - size) % size}px 0`; // wrap into one copy
  requestAnimationFrame(frame);
}

scroller.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  scroller.setPointerCapture(e.pointerId);
  pressed = true;
  dragged = false;
  startP = pointer(e);
  lastP = startP;
  state = 'drag'; // pressing takes over from autoplay and momentum
  v = 0;
});

scroller.addEventListener('pointermove', (e) => {
  if (!pressed) return;
  const p = pointer(e);
  x += p - lastP;
  lastP = p;
  dragged = dragged || Math.abs(p - startP) > 2;
});

function release(e) {
  if (!pressed) return;
  pressed = false;
  if (state === 'drag') {
    snapTo(x);
    return;
  }
  letGo();
}

scroller.addEventListener('pointerup', release);
scroller.addEventListener('pointercancel', release);
scroller.addEventListener(
  'click',
  (e) => {
    if (!dragged) return;
    e.preventDefault(); // a drag is not a click on a card
    e.stopPropagation();
  },
  true,
);
scroller.addEventListener('dragstart', (e) => e.preventDefault());

scroller.addEventListener('pointerenter', (e) => {
  if (e.pointerType !== 'mouse') return; // touch has no hover
  hovered = true;
  if (state === 'autoplay' || state === 'resume') {
    start('decel', v, 0, PAUSE_DECEL_MS, PAUSE_EASE);
  }
});
scroller.addEventListener('pointerleave', (e) => {
  if (e.pointerType !== 'mouse') return;
  hovered = false;
  if (!pressed) letGo();
});

// requestAnimationFrame stops while the tab is hidden: restart the clock so nothing jumps
document.addEventListener('visibilitychange', () => {
  last = performance.now();
});
requestAnimationFrame(frame);
