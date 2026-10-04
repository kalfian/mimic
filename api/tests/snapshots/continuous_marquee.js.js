// Suggested driver — estimated from a screen recording, not the original code.
// All values are approximate (display scale assumed 1x).
// Markup: .scroller > .scroller__track holding the cards twice (two identical copies back to back).
// Signed velocities: positive = content moves right (x) / down (y).

const AUTOPLAY_PX_S = -240; // linear autoplay to the left, high confidence

const scroller = document.querySelector('.scroller');
const track = scroller.querySelector('.scroller__track');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

let x = 0; // track offset in px, unwrapped (wrapped only when rendering)
let v = 0; // velocity in px/s
let autoV = AUTOPLAY_PX_S; // current autoplay velocity
let state = 'autoplay'; // autoplay | decel | paused | drag | inertia | snap | waiting | resume
let last = performance.now();

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

function frame(now) {
  const dt = (now - last) / 1000;
  last = now;
  if (state === 'autoplay') {
    v = canAutoplay() ? autoV : 0;
  }
  x += v * dt; // v is zero while dragging, snapping, gliding or paused
  const size = copySize();
  if (size > 0) track.style.translate = `${((x % size) - size) % size}px 0`; // wrap into one copy
  requestAnimationFrame(frame);
}

// requestAnimationFrame stops while the tab is hidden: restart the clock so nothing jumps
document.addEventListener('visibilitychange', () => {
  last = performance.now();
});
requestAnimationFrame(frame);
