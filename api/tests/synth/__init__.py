"""Synthetic UI-recording generator with exact ground truth (PLAN §11.2, Track S).

Modules: ``animate`` (independent cubic-bezier + tweens + cursor path), ``scene`` (node tree +
compositor), ``cursor_sprite``, ``scenarios`` (S1–S12 + truth derivation), ``truth`` (truth file
models), ``encode`` (ffmpeg variants + ffprobe). Render everything with
``uv run python scripts/make_synth.py`` (``make synth``).
"""
