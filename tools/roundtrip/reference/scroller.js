// Reference scroller driver for the round-trip harness self-check (PLAN-continuous §14).
//
// Hand-written from the synthetic ground truth (api/tests/synth/kinematics.py + scenarios.py),
// NOT from Mimic's outputs: it is the "known-correct implementation" the harness must accept.
// One requestAnimationFrame loop owns the content position x (CSS px, + = content moves right /
// down, the IR sign convention); the track is translated by wrap(x) over two identical copies.
//
// States: autoplay -> (hover) decel -> paused -> (leave) wait -> resume -> autoplay
//         autoplay|any -> (press) drag -> (release) inertia | snap | rest -> wait -> resume
// Time comes from performance.now() / event.timeStamp; the simulation integrates in <= 1 ms
// sub-steps so phase boundaries land on the millisecond, independent of the frame rate.
//
// cfg = {
//   axis: 'x' | 'y', autoplay: px/s (signed), period: px (one content copy),
//   hoverPause: null | { decelMs, ease: [x1, y1, x2, y2] },        // slow to 0 while hovered
//   drag: boolean, releaseWindowMs: 50,                              // release velocity window
//   inertia: null | { tauMs, stopPxS },                              // v *= exp(-dt / tau)
//   snap: null | { step, minPx, ms, ease },                          // tween to next grid point
//   resume: null | { delayAfterRestMs, delayAfterLeaveMs, rampMs, ease },
// }
(function () {
  'use strict';

  function bezier(x1, y1, x2, y2) {
    const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
    const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
    const sx = (t) => ((ax * t + bx) * t + cx) * t;
    const sy = (t) => ((ay * t + by) * t + cy) * t;
    const dx = (t) => (3 * ax * t + 2 * bx) * t + cx;
    return function (p) {
      if (p <= 0) return 0;
      if (p >= 1) return 1;
      let t = p;
      for (let i = 0; i < 8; i++) {
        const e = sx(t) - p;
        const d = dx(t);
        if (Math.abs(e) < 1e-7) return sy(t);
        if (Math.abs(d) < 1e-6) break;
        t -= e / d;
      }
      let lo = 0, hi = 1;
      t = p;
      for (let i = 0; i < 40; i++) {
        const e = sx(t) - p;
        if (Math.abs(e) < 1e-7) break;
        if (e > 0) hi = t; else lo = t;
        t = (lo + hi) / 2;
      }
      return sy(t);
    };
  }

  function mountScroller(root, cfg) {
    const track = root.querySelector('.track');
    const axis = cfg.axis || 'x';
    const period = cfg.period;
    const vAuto = cfg.autoplay || 0;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const decelEase = cfg.hoverPause ? bezier(...cfg.hoverPause.ease) : null;
    const rampEase = cfg.resume ? bezier(...cfg.resume.ease) : null;
    const snapEase = cfg.snap ? bezier(...cfg.snap.ease) : null;

    let x = 0;
    let v = reduced ? 0 : vAuto;
    let state = reduced ? 'rest' : 'autoplay';
    let tState = 0; // start of the current state (ms)
    let vFrom = 0; // velocity when a ramp started
    let xFrom = 0, xTo = 0; // snap tween
    let restAt = 0; // when the content came to rest
    let leftAt = null; // pointer left the scroller at
    let hovered = false;
    let tSim = null; // simulation clock (ms)
    let drag = null; // { id, p0, x0, samples: [[t, p]] }

    const pointerPos = (e) => (axis === 'x' ? e.clientX : e.clientY);

    function render() {
      const w = ((x % period) + period) % period - period;
      track.style.transform = axis === 'x' ? `translate3d(${w}px,0,0)` : `translate3d(0,${w}px,0)`;
    }

    function waitingDue() {
      // when the resume ramp should start (ms), or null while it must not
      if (!cfg.resume || reduced) return null;
      if (cfg.hoverPause && hovered) return null;
      const candidates = [restAt + cfg.resume.delayAfterRestMs];
      if (cfg.hoverPause && leftAt !== null && cfg.resume.delayAfterLeaveMs != null) {
        candidates.push(leftAt + cfg.resume.delayAfterLeaveMs);
      }
      return Math.max(...candidates);
    }

    function enter(s, t) {
      state = s;
      tState = t;
    }

    // advance the simulation from tSim to t in <= 1 ms sub-steps
    function step(t) {
      if (tSim === null) { tSim = t; return; }
      while (tSim < t - 1e-9) {
        const h = Math.min(1, t - tSim);
        const t1 = tSim + h;
        switch (state) {
          case 'autoplay':
            v = vAuto;
            x += v * h / 1000;
            if (cfg.hoverPause && hovered) { vFrom = v; enter('decel', t1); }
            break;
          case 'decel': {
            const D = cfg.hoverPause.decelMs;
            const p = Math.min(1, (t1 - tState) / D);
            const v1 = vFrom * (1 - decelEase(p));
            x += (v + v1) / 2 * h / 1000;
            v = v1;
            if (p >= 1) { v = 0; restAt = t1; enter('rest', t1); }
            else if (!hovered) { vFrom = v; enter('resume_from', t1); }
            break;
          }
          case 'resume_from': // left during the slowdown: ramp back up from the current speed
          case 'resume': {
            const D = cfg.resume ? cfg.resume.rampMs : 1;
            const p = Math.min(1, (t1 - tState) / D);
            const v1 = vFrom + (vAuto - vFrom) * (rampEase ? rampEase(p) : 1);
            x += (v + v1) / 2 * h / 1000;
            v = v1;
            if (p >= 1) { v = vAuto; enter('autoplay', t1); }
            else if (cfg.hoverPause && hovered) { vFrom = v; enter('decel', t1); }
            break;
          }
          case 'inertia': {
            const tau = cfg.inertia.tauMs;
            const stop = cfg.inertia.stopPxS;
            const k = Math.exp(-h / tau);
            const v1 = v * k;
            if (Math.abs(v1) <= stop) {
              // exact crossing time of |v| = stop inside this sub-step
              const u = tau * Math.log(Math.abs(v) / stop);
              x += v * tau * (1 - Math.exp(-u / tau)) / 1000;
              v = 0;
              restAt = tSim + u;
              enter('rest', restAt);
            } else {
              x += v * tau * (1 - k) / 1000;
              v = v1;
            }
            break;
          }
          case 'snap': {
            const p = Math.min(1, (t1 - tState) / cfg.snap.ms);
            const xn = xFrom + (xTo - xFrom) * snapEase(p);
            v = (xn - x) / (h / 1000);
            x = xn;
            if (p >= 1) { x = xTo; v = 0; restAt = t1; enter('rest', t1); }
            break;
          }
          case 'rest': {
            v = 0;
            const due = waitingDue();
            if (due !== null && t1 >= due) { vFrom = 0; enter('resume', due); }
            break;
          }
          case 'drag':
          default:
            break;
        }
        tSim = t1;
      }
    }

    function frame(ts) {
      step(ts);
      render();
      requestAnimationFrame(frame);
    }

    root.addEventListener('pointerenter', (e) => {
      if (e.pointerType !== 'mouse') return;
      step(e.timeStamp);
      hovered = true;
      leftAt = null;
    });
    root.addEventListener('pointerleave', (e) => {
      if (e.pointerType !== 'mouse') return;
      step(e.timeStamp);
      hovered = false;
      leftAt = e.timeStamp;
    });

    if (cfg.drag) {
      root.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        step(e.timeStamp);
        root.setPointerCapture(e.pointerId);
        const p = pointerPos(e);
        drag = { id: e.pointerId, p0: p, x0: x, samples: [[e.timeStamp, p]] };
        v = 0; // a press stops the content at once
        enter('drag', e.timeStamp);
        root.classList.add('is-dragging');
        e.preventDefault();
      });
      root.addEventListener('pointermove', (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        step(e.timeStamp);
        const p = pointerPos(e);
        x = drag.x0 + (p - drag.p0);
        drag.samples.push([e.timeStamp, p]);
        render();
      });
      const release = (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        step(e.timeStamp);
        const t = e.timeStamp;
        const p = pointerPos(e);
        x = drag.x0 + (p - drag.p0);
        drag.samples.push([t, p]);
        // release velocity: displacement over the last releaseWindowMs of pointer samples
        const win = cfg.releaseWindowMs || 50;
        const s = drag.samples;
        const last = s[s.length - 1];
        let first = last;
        for (let i = s.length - 1; i >= 0; i--) {
          if (last[0] - s[i][0] > win + 1e-6) break;
          first = s[i];
        }
        const dt = last[0] - first[0];
        const vr = dt > 0 ? ((last[1] - first[1]) / dt) * 1000 : 0;
        drag = null;
        root.classList.remove('is-dragging');
        if (cfg.snap && vr !== 0) {
          const st = cfg.snap.step;
          const k = vr > 0 ? Math.ceil((x + cfg.snap.minPx) / st) : Math.floor((x - cfg.snap.minPx) / st);
          xFrom = x;
          xTo = k * st;
          enter('snap', t);
        } else if (cfg.inertia && Math.abs(vr) > cfg.inertia.stopPxS) {
          v = vr;
          enter('inertia', t);
        } else {
          v = 0;
          restAt = t;
          enter('rest', t);
        }
      };
      root.addEventListener('pointerup', release);
      root.addEventListener('pointercancel', release);
    }

    render();
    requestAnimationFrame(frame);
  }

  function buildCards(track, n, copies) {
    const frag = document.createDocumentFragment();
    for (let c = 0; c < copies; c++) {
      for (let k = 0; k < n; k++) {
        const h1 = (k * 47 + 12) % 360, h2 = (k * 47 + 140) % 360, h3 = (k * 47 + 250) % 360;
        const card = document.createElement('div');
        card.className = 'card';
        const img = document.createElement('div');
        img.className = 'card__img';
        img.style.background =
          `radial-gradient(circle at ${20 + (k * 13) % 60}% ${30 + (k * 29) % 40}%, hsl(${h3} 70% 55%) 0 14%, transparent 15%),` +
          `radial-gradient(circle at ${70 - (k * 11) % 40}% ${65 - (k * 7) % 30}%, hsl(${h2} 60% 40%) 0 10%, transparent 11%),` +
          `repeating-linear-gradient(${(k * 37) % 180}deg, hsl(${h1} 55% 62%) 0 7px, hsl(${h1} 45% 48%) 7px 13px)`;
        const title = document.createElement('div');
        title.className = 'card__title';
        title.textContent = `Item ${k + 1}`;
        const sub = document.createElement('div');
        sub.className = 'card__sub';
        sub.textContent = 'Placeholder';
        card.append(img, title, sub);
        frag.append(card);
      }
    }
    track.append(frag);
  }

  window.RefScroller = { mountScroller, buildCards, bezier };
})();
