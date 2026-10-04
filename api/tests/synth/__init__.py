"""Synthetic UI-recording generator with exact ground truth (PLAN §11.2, Track S;
PLAN-continuous §8).

Modules: ``animate`` (independent cubic-bezier + tweens + cursor path), ``kinematics``
(velocity profiles for scrollers: autoplay, ramps, drag, inertia, snap; own integrator),
``scene`` (node tree + compositor, kinematic drivers), ``cursor_sprite``, ``scenarios``
(S1–S12, C1–C12, N1 + truth derivation), ``truth`` (truth file models), ``encode`` (ffmpeg
variants incl. VFR and the recorder capture grid + ffprobe), ``evaluate`` (IR vs truth,
§11.4 / §8.4 targets). Render everything with ``uv run python scripts/make_synth.py``
(``make synth``); judge with ``make eval``.
"""
