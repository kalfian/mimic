"""Compare synthetic-suite IR files against a saved baseline (PLAN-continuous §9 P0 / §8.4).

    # after `make eval`: compare data/synth/<name>.ir.json with data/synth/baseline/<name>.ir.json
    uv run python scripts/compare_ir_baseline.py

    # explicit directories / a subset
    uv run python scripts/compare_ir_baseline.py --current ../data/synth \\
        --baseline ../data/synth/baseline accordion_expand button_press

Regression rule: apart from the ignored fields, every IR must be identical to the baseline.
Ignored anywhere in the document (spec, envelope and the serialized ``outputs.json`` export):

- ``schema_version`` and ``mode`` (IR 0.1 -> 0.2 adds ``mode``; transition specs otherwise equal)
- ``meta.generated_at`` and ``meta.pipeline_version``
- ``outputs.js`` when it is ``null`` / absent (IR 0.2 adds the optional JS output; transition
  mode never fills it)

Text outputs (``technical`` / ``llm_prompt`` / ``css``) are compared verbatim. ``outputs.json`` is
parsed and compared structurally with the same ignore rules.

Measured appearance (PLAN-continuous §13, wired in P2) is an intended addition to transition
IRs: ``scene`` and ``ElementStatic.background_color`` / ``text_color`` / ``font_size_px``. By
default (``--appearance strip``) the current spec is validated, those fields are removed, and the
four outputs are **re-rendered from the stripped spec** before comparing, so a clean result
proves that everything except the appearance additions (and the text the generators derive
from them) is byte-identical. The rendered outputs of the unstripped spec must also equal the
stored ones. ``--appearance keep`` compares the files as they are.

Exit: 0 clean, 1 differences found, 2 nothing to compare / missing files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parent
DEFAULT_CURRENT = REPO_ROOT / "data" / "synth"
DEFAULT_BASELINE = DEFAULT_CURRENT / "baseline"

IGNORED_KEYS = frozenset({"schema_version", "mode"})
IGNORED_META_KEYS = frozenset({"generated_at", "pipeline_version"})
MAX_DIFFS_PER_FILE = 20


def normalize(node: Any, path: tuple[str, ...] = ()) -> Any:
    """Drop ignored fields recursively; parse the embedded JSON export so it is compared too."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in IGNORED_KEYS:
                continue
            if path and path[-1] == "meta" and key in IGNORED_META_KEYS:
                continue
            if path and path[-1] == "outputs":
                if key == "js" and value is None:
                    continue
                if key == "json" and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # compare the raw string
            out[key] = normalize(value, (*path, key))
        return out
    if isinstance(node, list):
        return [normalize(v, (*path, f"[{i}]")) for i, v in enumerate(node)]
    return node


def diff(a: Any, b: Any, path: str = "$") -> list[str]:
    """Structural differences between baseline ``a`` and current ``b`` (paths, short values)."""
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} -> {type(b).__name__}"]
    if isinstance(a, dict):
        out: list[str] = []
        for key in sorted(a.keys() | b.keys()):
            if key not in b:
                out.append(f"{path}.{key}: removed")
            elif key not in a:
                out.append(f"{path}.{key}: added ({_short(b[key])})")
            else:
                out.extend(diff(a[key], b[key], f"{path}.{key}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} -> {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(diff(x, y, f"{path}[{i}]"))
        return out
    if a != b:
        return [f"{path}: {_short(a)} -> {_short(b)}"]
    return []


def _short(v: Any, limit: int = 80) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 3] + "..."


APPEARANCE_STATIC_KEYS = ("background_color", "text_color", "font_size_px")


def strip_appearance(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``data`` with the appearance additions removed and the outputs re-rendered from the
    stripped spec. Returns ``(data, notes)``; notes list what was stripped (or a consistency
    failure of the stored outputs)."""
    if "spec" not in data:
        return data, []
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))
    from app.generate import render_all
    from app.models.ir import ElementStatic, MotionSpec

    spec = MotionSpec.model_validate(data["spec"])
    notes: list[str] = []
    if "outputs" in data:
        stored = data["outputs"]
        rendered = render_all(spec).model_dump(mode="json")
        for key in ("technical", "llm_prompt", "css"):
            if stored.get(key) != rendered.get(key):
                notes.append(f"outputs.{key} != render_all(spec) (stale outputs)")
    elements = []
    for el in spec.elements:
        st = el.static
        added = [k for k in APPEARANCE_STATIC_KEYS if getattr(st, k) is not None]
        if added:
            notes.append(f"{el.id}: {', '.join(added)}")
            st = ElementStatic(border_radius_px=st.border_radius_px, shadow=st.shadow)
        elements.append(el.model_copy(update={"static": st}))
    if spec.scene is not None:
        notes.append("scene")
    stripped = spec.model_copy(update={"scene": None, "elements": elements})
    out = dict(data)
    out["spec"] = json.loads(stripped.model_dump_json())
    if "outputs" in data:
        out["outputs"] = render_all(stripped).model_dump(mode="json")
    return out, notes


def compare_files(baseline: Path, current: Path, appearance: str = "strip") -> list[str]:
    a = normalize(json.loads(baseline.read_text(encoding="utf-8")))
    cur = json.loads(current.read_text(encoding="utf-8"))
    notes: list[str] = []
    if appearance == "strip":
        cur, notes = strip_appearance(cur)
    b = normalize(cur)
    bad = [n for n in notes if "stale outputs" in n]
    return bad + diff(a, b)


def appearance_notes(current: Path) -> list[str]:
    return strip_appearance(json.loads(current.read_text(encoding="utf-8")))[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (default: every baseline file)")
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument(
        "--appearance",
        choices=["strip", "keep"],
        default="strip",
        help="strip the §13 appearance additions before comparing (default) or keep them",
    )
    args = ap.parse_args(argv)

    if args.names:
        base_files = [args.baseline / f"{n}.ir.json" for n in args.names]
    else:
        base_files = sorted(args.baseline.glob("*.ir.json"))
    if not base_files:
        print(f"no baseline IR files in {args.baseline}", file=sys.stderr)
        return 2

    missing, dirty = [], 0
    for bf in base_files:
        cf = args.current / bf.name
        if not bf.exists() or not cf.exists():
            missing.append(bf.name)
            continue
        diffs = compare_files(bf, cf, args.appearance)
        name = bf.name.removesuffix(".ir.json")
        if diffs:
            dirty += 1
            print(f"DIFF  {name}: {len(diffs)} difference(s)")
            for d in diffs[:MAX_DIFFS_PER_FILE]:
                print(f"      {d}")
            if len(diffs) > MAX_DIFFS_PER_FILE:
                print(f"      ... {len(diffs) - MAX_DIFFS_PER_FILE} more")
        else:
            extra = ""
            if args.appearance == "strip":
                notes = appearance_notes(cf)
                extra = f"  (appearance stripped: {'; '.join(notes)})" if notes else ""
            print(f"ok    {name}{extra}")

    compared = len(base_files) - len(missing)
    if missing:
        print(f"missing: {', '.join(missing)}")
    print(f"baseline compare: {compared - dirty}/{compared} clean")
    if missing or compared == 0:
        return 2
    return 1 if dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
